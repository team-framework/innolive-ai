#!/usr/bin/env bash
# AI 워커를 하나씩 새 릴리스로 바꾼다. 진행 중인 방송은 끊기지 않는다.
#
# 워커마다 새 릴리스로 다른 포트에 새 인스턴스를 띄우고, Go 미디어 서버의 제어 API
# (POST /ai/workers/replace)로 옛 워커를 대신하게 한다. 서버는 화이트리스트를 옮기고
# 진행 중인 세션을 다음 프레임부터 새 워커로 넘긴 뒤 옛 워커를 풀에서 뺀다. 그 다음에야
# 옛 인스턴스를 멈춘다. 새 워커 하나 몫의 GPU 메모리만 더 쓴다.
#
# 인스턴스는 슬롯 두 개를 번갈아 쓴다. innolive-ai@<n>(포트 P)과 innolive-ai@<n>b
# (포트 P±100). 지금 도는 슬롯의 drop-in·env 파일을 복사해 릴리스 경로와 포트만 바꾼다.
#
# 사용법(root):
#   deploy/rolling_replace.sh /opt/innolive/ai-releases/<rev> [인스턴스 번호...]
#
# 환경 변수:
#   CONTROL_URL      제어 API 주소 (기본 http://127.0.0.1:9091)
#   SERVER_ENV       AI_CONTROL_TOKEN·AI_GRPC_TARGETS를 읽고 갱신할 서버 env 파일
#                    (기본 /etc/innolive/server.env)
#   ALLOW_REENROLL   1이면 화이트리스트를 옮기지 못해도 진행한다. 등록 얼굴은 재등록
#                    필요로 표시된다. 화이트리스트 이동 RPC가 없는 릴리스에서 처음 바꿀
#                    때만 쓴다.
#   MIN_FREE_MIB     새 인스턴스를 띄우기 전 GPU에 남아 있어야 할 메모리 (기본 4096)
#   READY_TIMEOUT    새 인스턴스가 gRPC SERVING이 되길 기다리는 초 (기본 180)
#   DRAIN_RETRIES    옛 워커에 세션이 남았을 때 교체 요청을 다시 보내는 횟수 (기본 5)
set -euo pipefail

RELEASE_DIR=${1:?usage: rolling_replace.sh RELEASE_DIR [INSTANCE...]}
shift
INSTANCES=("$@")
[ ${#INSTANCES[@]} -gt 0 ] || INSTANCES=(0 1 2 3 4 5)

CONTROL_URL=${CONTROL_URL:-http://127.0.0.1:9091}
SERVER_ENV=${SERVER_ENV:-/etc/innolive/server.env}
ALLOW_REENROLL=${ALLOW_REENROLL:-0}
MIN_FREE_MIB=${MIN_FREE_MIB:-4096}
READY_TIMEOUT=${READY_TIMEOUT:-180}
DRAIN_RETRIES=${DRAIN_RETRIES:-5}
UNIT_DIR=/etc/systemd/system
ENV_DIR=/etc/innolive

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

[ "$(id -u)" -eq 0 ] || die "run as root"
[ -x "$RELEASE_DIR/venv/bin/python" ] || die "$RELEASE_DIR/venv/bin/python not found"
[ -f "$RELEASE_DIR/ai_processor_server.py" ] || die "$RELEASE_DIR is not an AI release"
CONTROL_TOKEN=$(sed -n 's/^AI_CONTROL_TOKEN=//p' "$SERVER_ENV" | tail -1)
[ -n "$CONTROL_TOKEN" ] || die "AI_CONTROL_TOKEN is not set in $SERVER_ENV"

control() {
	# control METHOD PATH [BODY] -> 본문을 stdout, HTTP 코드를 마지막 줄로
	local method=$1 path=$2 body=${3:-}
	curl -sS -X "$method" -H "Authorization: Bearer $CONTROL_TOKEN" \
		-H 'Content-Type: application/json' ${body:+--data "$body"} \
		--max-time 360 -w '\n%{http_code}' "$CONTROL_URL$path"
}

# active_slot N -> 지금 도는 인스턴스 이름(N 또는 Nb)
active_slot() {
	local n=$1
	if systemctl is-active --quiet "innolive-ai@${n}b"; then echo "${n}b"
	elif systemctl is-active --quiet "innolive-ai@${n}"; then echo "$n"
	else return 1
	fi
}

other_slot() { case $1 in *b) echo "${1%b}" ;; *) echo "${1}b" ;; esac; }

env_value() { sed -n "s/^$2=//p" "$1" | tail -1; }

# env 파일 경로 목록: 슬롯 유닛이 읽는 EnvironmentFile 중 /etc/innolive 아래 것
slot_env_files() {
	systemctl show -p EnvironmentFiles "innolive-ai@$1" | grep -o "$ENV_DIR/[^ ]*-$1\.env" | sort -u
}

gpu_free_mib() {
	local device=$1
	nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$device" | tr -d ' '
}

wait_serving() {
	local port=$1 deadline=$((SECONDS + READY_TIMEOUT))
	while [ $SECONDS -lt $deadline ]; do
		if "$RELEASE_DIR/venv/bin/python" -I - "$port" <<'PY' 2>/dev/null; then
import sys, grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
channel = grpc.insecure_channel(f"127.0.0.1:{sys.argv[1]}")
status = health_pb2_grpc.HealthStub(channel).Check(health_pb2.HealthCheckRequest(), timeout=3).status
sys.exit(0 if status == health_pb2.HealthCheckResponse.SERVING else 1)
PY
			return 0
		fi
		sleep 3
	done
	return 1
}

# prepare_slot OLD NEW NEW_PORT -> 새 슬롯의 env·drop-in을 옛 슬롯에서 만든다
prepare_slot() {
	local old=$1 new=$2 port=$3 file target
	for file in $(slot_env_files "$old"); do
		target=${file%-"$old".env}-"$new".env
		sed -e "s/^GRPC_PORT=.*/GRPC_PORT=$port/" "$file" >"$target"
		chmod --reference="$file" "$target"
	done
	rm -rf "$UNIT_DIR/innolive-ai@$new.service.d"
	if [ -d "$UNIT_DIR/innolive-ai@$old.service.d" ]; then
		cp -a "$UNIT_DIR/innolive-ai@$old.service.d" "$UNIT_DIR/innolive-ai@$new.service.d"
		# 릴리스 경로와 슬롯별 env 파일 이름만 바꾼다.
		sed -i -E \
			-e "s#/opt/innolive/ai-releases/[^/ ]+#$RELEASE_DIR#g" \
			-e "s#($ENV_DIR/[^ ]*)-$old\\.env#\\1-$new.env#g" \
			"$UNIT_DIR/innolive-ai@$new.service.d/"*.conf
	fi
	systemctl daemon-reload
}

cleanup_slot() {
	local slot=$1 file
	systemctl stop "innolive-ai@$slot" 2>/dev/null || true
	for file in $(slot_env_files "$slot" 2>/dev/null); do rm -f "$file"; done
	rm -rf "$UNIT_DIR/innolive-ai@$slot.service.d"
	systemctl daemon-reload
}

# 서버가 다시 시작해도 바뀐 워커 목록을 쓰도록 env를 고친다.
persist_target() {
	local from=$1 to=$2
	cp -a "$SERVER_ENV" "$SERVER_ENV.bak-$(date +%Y%m%d%H%M%S)"
	sed -i -E "/^AI_(GRPC|FACE_SWAP)_TARGETS=/ s#([=,])$from(,|\$)#\\1$to\\2#g" "$SERVER_ENV"
}

replace_instance() {
	local n=$1 old new old_port new_port device body response code attempt
	old=$(active_slot "$n") || die "innolive-ai@$n is not running"
	new=$(other_slot "$old")
	old_port=$(env_value "$(slot_env_files "$old" | head -1)" GRPC_PORT)
	[ -n "$old_port" ] || die "GRPC_PORT not found for innolive-ai@$old"
	if [ "$old_port" -lt 50100 ]; then new_port=$((old_port + 100)); else new_port=$((old_port - 100)); fi
	# 인스턴스는 env의 CUDA_VISIBLE_DEVICES(GPU UUID)로 카드에 고정된다. 없으면 --device.
	device=$(env_value "$(slot_env_files "$old" | head -1)" CUDA_VISIBLE_DEVICES)
	if [ -z "$device" ]; then
		device=$(systemctl show -p ExecStart "innolive-ai@$old" | grep -oE -- '--device [0-9]+' | awk '{print $2}' | head -1)
		device=${device:-0}
	fi

	local free
	free=$(gpu_free_mib "$device")
	[ "$free" -ge "$MIN_FREE_MIB" ] || die "GPU $device has ${free}MiB free, need ${MIN_FREE_MIB}MiB for innolive-ai@$new"

	log "innolive-ai@$old (:$old_port) -> innolive-ai@$new (:$new_port) on GPU $device"
	prepare_slot "$old" "$new" "$new_port"
	systemctl start "innolive-ai@$new"
	if ! wait_serving "$new_port"; then
		cleanup_slot "$new"
		die "innolive-ai@$new did not become SERVING within ${READY_TIMEOUT}s"
	fi

	body=$(printf '{"from":"127.0.0.1:%s","to":"127.0.0.1:%s","allow_reenrollment":%s}' \
		"$old_port" "$new_port" "$([ "$ALLOW_REENROLL" = 1 ] && echo true || echo false)")
	for attempt in $(seq 0 "$DRAIN_RETRIES"); do
		response=$(control POST /ai/workers/replace "$body") || true
		code=$(tail -n1 <<<"$response")
		log "replace attempt $attempt: HTTP $code $(sed '$d' <<<"$response")"
		case $code in
		200) break ;;
		202) sleep 10 ;;
		*)
			cleanup_slot "$new"
			die "server refused the replacement; innolive-ai@$old keeps serving"
			;;
		esac
	done
	if [ "$code" != 200 ]; then
		die "sessions remain on innolive-ai@$old; both instances left running. Rerun after they end."
	fi

	persist_target "127.0.0.1:$old_port" "127.0.0.1:$new_port"
	systemctl enable --quiet "innolive-ai@$new"
	systemctl disable --quiet --now "innolive-ai@$old"
	log "innolive-ai@$new replaced innolive-ai@$old"
}

for n in "${INSTANCES[@]}"; do
	replace_instance "$n"
done
control GET /ai/workers | sed '$d'
log "done: ${INSTANCES[*]} now run $RELEASE_DIR"
