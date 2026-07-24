#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
VERSION_FILE="$ROOT_DIR/benchmarks/ghz.version"
INSTALL_DIR="${GHZ_INSTALL_DIR:-$ROOT_DIR/.benchmark-tools/bin}"
VERSION="${GHZ_VERSION:-$(tr -d '[:space:]' < "$VERSION_FILE")}" 

if [ -x "$INSTALL_DIR/ghz" ]; then
  installed_version=$("$INSTALL_DIR/ghz" --version 2>&1 | tr -d '[:space:]')
  if [ "$installed_version" = "$VERSION" ]; then
    echo "ghz ${VERSION} is already installed at $INSTALL_DIR/ghz"
    exit 0
  fi
fi

case "$(uname -s)" in
  Darwin) os=darwin ;;
  Linux) os=linux ;;
  *) echo "Unsupported operating system: $(uname -s)" >&2; exit 1 ;;
esac

case "$(uname -m)" in
  arm64|aarch64) arch=arm64 ;;
  x86_64|amd64) arch=x86_64 ;;
  *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

archive="ghz-${os}-${arch}.tar.gz"
base_url="https://github.com/bojand/ghz/releases/download/${VERSION}"
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/ghz.XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT INT TERM

echo "Installing ghz ${VERSION} for ${os}/${arch}"
curl -fsSL "$base_url/$archive" -o "$tmp_dir/$archive"
curl -fsSL "$base_url/$archive.sha256" -o "$tmp_dir/$archive.sha256"

expected=$(awk '{print $1}' "$tmp_dir/$archive.sha256")
if command -v shasum >/dev/null 2>&1; then
  actual=$(shasum -a 256 "$tmp_dir/$archive" | awk '{print $1}')
elif command -v sha256sum >/dev/null 2>&1; then
  actual=$(sha256sum "$tmp_dir/$archive" | awk '{print $1}')
else
  echo "Neither shasum nor sha256sum is available" >&2
  exit 1
fi

if [ "$expected" != "$actual" ]; then
  echo "Checksum verification failed for $archive" >&2
  exit 1
fi

mkdir -p "$INSTALL_DIR"
tar -xzf "$tmp_dir/$archive" -C "$tmp_dir"
install -m 0755 "$tmp_dir/ghz" "$INSTALL_DIR/ghz"
"$INSTALL_DIR/ghz" --version
