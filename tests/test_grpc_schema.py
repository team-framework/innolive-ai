from __future__ import annotations

import unittest
from unittest.mock import Mock

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from protos import ai_processor_pb2, ai_processor_pb2_grpc


class GrpcSchemaContractTests(unittest.TestCase):
    def test_process_video_keeps_the_legacy_unqualified_bidi_route(self):
        service = ai_processor_pb2.DESCRIPTOR.services_by_name["AiProcessor"]
        method = service.methods_by_name["ProcessVideo"]

        self.assertEqual(service.full_name, "AiProcessor")
        self.assertTrue(method.client_streaming)
        self.assertTrue(method.server_streaming)
        self.assertEqual(method.input_type.full_name, "VideoChunk")
        self.assertEqual(method.output_type.full_name, "ProcessedVideoChunk")

        channel = Mock()
        ai_processor_pb2_grpc.AiProcessorStub(channel)
        self.assertEqual(
            channel.stream_stream.call_args.args[0],
            "/AiProcessor/ProcessVideo",
        )

    def test_every_legacy_field_number_and_type_is_stable(self):
        expected = {
            "VideoChunk": {
                "data": (1, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES),
                "timestamp": (2, descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
                "frame_id": (3, descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
                "batch_size": (4, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32),
            },
            "MosaicConfig": {
                "blur_radius": (1, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE),
                "pixel_size": (2, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32),
            },
            "ProcessedVideoChunk": {
                "data": (1, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES),
                "timestamp": (2, descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
                "status_message": (
                    3,
                    descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
                ),
                "faces": (4, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE),
                "width": (5, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
                "height": (6, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
                "frame_id": (7, descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
                "processing_ms": (
                    8,
                    descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE,
                ),
                "timing": (9, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE),
            },
            "ProcessingTiming": {
                "queue_ms": (1, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE),
                "decode_ms": (2, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE),
                "inference_ms": (3, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE),
                "tracking_ms": (4, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE),
                "blur_encode_ms": (
                    5,
                    descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE,
                ),
                "inference_batch_size": (
                    6,
                    descriptor_pb2.FieldDescriptorProto.TYPE_UINT32,
                ),
            },
            "FaceMetadata": {
                "bbox": (1, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE),
                "confidence": (2, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT),
                "polygon": (3, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE),
                "track_id": (4, descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
            },
        }

        for message_name, fields in expected.items():
            descriptor = ai_processor_pb2.DESCRIPTOR.message_types_by_name[message_name]
            actual = {
                name: (field.number, field.type)
                for name, field in descriptor.fields_by_name.items()
                if name in fields
            }
            self.assertEqual(actual, fields)

    def test_optional_track_id_presence_and_deprecations_are_explicit(self):
        face = ai_processor_pb2.FaceMetadata()
        self.assertFalse(face.HasField("track_id"))
        face.track_id = 0
        self.assertTrue(face.HasField("track_id"))

        mosaic = ai_processor_pb2.MosaicConfig()
        self.assertFalse(mosaic.HasField("blur_radius"))
        self.assertFalse(mosaic.HasField("pixel_size"))
        self.assertFalse(mosaic.HasField("mask_margin"))
        mosaic.blur_radius = 24.0
        mosaic.pixel_size = 2
        mosaic.mask_margin = 0
        self.assertTrue(mosaic.HasField("blur_radius"))
        self.assertTrue(mosaic.HasField("pixel_size"))
        parsed = ai_processor_pb2.MosaicConfig.FromString(mosaic.SerializeToString())
        self.assertTrue(parsed.HasField("mask_margin"))
        self.assertEqual(parsed.mask_margin, 0)

        video = ai_processor_pb2.VideoChunk.DESCRIPTOR
        processed = ai_processor_pb2.ProcessedVideoChunk.DESCRIPTOR
        self.assertTrue(video.fields_by_name["batch_size"].GetOptions().deprecated)
        self.assertFalse(processed.fields_by_name["data"].GetOptions().deprecated)
        self.assertTrue(processed.fields_by_name["mosaic_jpeg"].GetOptions().deprecated)

    def test_session_and_whitelist_fields_are_additive(self):
        expected = {
            "VideoChunk": {
                "session_id": 5,
                "output_mode": 6,
                "mosaic_config": 7,
                "anonymization_mode": 11,
            },
            "MosaicConfig": {"blur_radius": 1, "pixel_size": 2, "mask_margin": 3},
            "ProcessedVideoChunk": {"mosaic_jpeg": 13},
            "FaceMetadata": {"whitelisted": 10, "anonymization": 11},
            "FrameStats": {
                "adaface_calls": 9,
                "adaface_queue_overflow": 10,
                "whitelisted_tracks": 11,
            },
            "FaceData": {"data": 1, "session_id": 2},
            "WhitelistResponse": {
                "status_message": 1,
                "timestamp": 2,
                "entry_id": 3,
                "entry_count": 4,
                "whitelist_version": 5,
            },
        }
        for message_name, fields in expected.items():
            descriptor = ai_processor_pb2.DESCRIPTOR.message_types_by_name[message_name]
            self.assertEqual(
                {name: descriptor.fields_by_name[name].number for name in fields},
                fields,
            )

        service = ai_processor_pb2.DESCRIPTOR.services_by_name["AiProcessor"]
        add_whitelist = service.methods_by_name["AddWhitelist"]
        self.assertFalse(add_whitelist.client_streaming)
        self.assertFalse(add_whitelist.server_streaming)
        get_status = service.methods_by_name["GetWhitelistStatus"]
        self.assertFalse(get_status.client_streaming)
        self.assertFalse(get_status.server_streaming)
        delete_whitelist = service.methods_by_name["DeleteWhitelist"]
        self.assertFalse(delete_whitelist.client_streaming)
        self.assertFalse(delete_whitelist.server_streaming)
        self.assertEqual(delete_whitelist.input_type.full_name, "DeleteWhitelistRequest")
        self.assertEqual(delete_whitelist.output_type.full_name, "WhitelistResponse")
        self.assertEqual(
            {
                field.name: field.number
                for field in ai_processor_pb2.DeleteWhitelistRequest.DESCRIPTOR.fields
            },
            {"session_id": 1, "entry_id": 2},
        )
        self.assertEqual(
            {
                field.name: field.number
                for field in ai_processor_pb2.GetWhitelistStatusResponse.DESCRIPTOR.fields
            },
            {
                "session_id": 1,
                "entry_count": 2,
                "whitelist_version": 3,
                "entry_ids": 4,
            },
        )
        for method_name in ("CreateSession", "ListSessions", "DeleteSession"):
            method = service.methods_by_name[method_name]
            self.assertFalse(method.client_streaming)
            self.assertFalse(method.server_streaming)
        delete_session = service.methods_by_name["DeleteSession"]
        self.assertEqual(delete_session.input_type.full_name, "DeleteSessionRequest")
        self.assertEqual(delete_session.output_type.full_name, "google.protobuf.Empty")
        self.assertEqual(
            ai_processor_pb2.DeleteSessionRequest.DESCRIPTOR.fields_by_name["session_id"].number,
            1,
        )
        self.assertEqual(
            {field.name: field.number for field in ai_processor_pb2.SessionInfo.DESCRIPTOR.fields},
            {
                "session_id": 1,
                "entry_count": 2,
                "whitelist_version": 3,
                "created_at_unix_ms": 4,
                "active_stream_count": 5,
            },
        )
        self.assertEqual(
            ai_processor_pb2.ListSessionsResponse.DESCRIPTOR.fields_by_name["sessions"].number,
            1,
        )

        output_modes = ai_processor_pb2.VideoOutputMode.DESCRIPTOR.values_by_name
        self.assertEqual(output_modes["VIDEO_OUTPUT_MODE_UNSPECIFIED"].number, 0)
        self.assertEqual(output_modes["VIDEO_OUTPUT_MODE_METADATA_ONLY"].number, 1)
        self.assertEqual(output_modes["VIDEO_OUTPUT_MODE_MOSAIC_JPEG"].number, 2)

    def test_vod_processor_service_is_additive_and_stable(self):
        service = ai_processor_pb2.DESCRIPTOR.services_by_name["VodProcessor"]
        self.assertEqual(service.full_name, "VodProcessor")
        streaming = {
            "StartVodJob": (False, False, "StartVodJobRequest", "VodJobStatus"),
            "WatchVodJob": (False, True, "VodJobRef", "VodJobStatus"),
            "GetVodJob": (False, False, "VodJobRef", "VodJobStatus"),
            "PauseVodJob": (False, False, "PauseVodJobRequest", "VodJobStatus"),
            "CancelVodJob": (False, False, "CancelVodJobRequest", "VodJobStatus"),
            "ListVodJobs": (False, False, "ListVodJobsRequest", "ListVodJobsResponse"),
        }
        for name, (client_stream, server_stream, request, response) in streaming.items():
            method = service.methods_by_name[name]
            self.assertEqual(method.client_streaming, client_stream)
            self.assertEqual(method.server_streaming, server_stream)
            self.assertEqual(method.input_type.full_name, request)
            self.assertEqual(method.output_type.full_name, response)

        channel = Mock()
        ai_processor_pb2_grpc.VodProcessorStub(channel)
        self.assertEqual(
            channel.unary_stream.call_args_list[0].args[0], "/VodProcessor/WatchVodJob"
        )

    def test_vod_messages_keep_their_field_numbers(self):
        status_fields = {
            field.name: field.number for field in ai_processor_pb2.VodJobStatus.DESCRIPTOR.fields
        }
        self.assertEqual(status_fields["job_id"], 1)
        self.assertEqual(status_fields["state"], 2)
        self.assertEqual(status_fields["checkpoint_token"], 4)
        self.assertEqual(status_fields["worker_target"], 15)
        self.assertEqual(status_fields["input_size_bytes"], 16)

        request = ai_processor_pb2.StartVodJobRequest.DESCRIPTOR
        self.assertEqual(request.fields_by_name["whitelist"].number, 5)
        self.assertEqual(request.fields_by_name["whitelist_session_id"].number, 6)
        self.assertEqual(request.fields_by_name["segment_frames"].number, 14)
        self.assertEqual(
            request.oneofs_by_name["whitelist_source"].name,
            "whitelist_source",
        )

        states = ai_processor_pb2.VodJobState.DESCRIPTOR.values_by_name
        self.assertEqual(states["VOD_JOB_STATE_PROCESSING"].number, 1)
        self.assertEqual(states["VOD_JOB_STATE_CANCELLED"].number, 5)
        self.assertEqual(ai_processor_pb2.VOD_QUALITY_HIGH, 1)
        pause = ai_processor_pb2.VodPauseReason.DESCRIPTOR.values_by_name
        self.assertEqual(pause["VOD_PAUSE_REASON_WORKER_DRAINING"].number, 2)
        self.assertEqual(pause["VOD_PAUSE_REASON_RUNTIME_RECOVERY"].number, 3)

    def test_mask_margin_is_compatible_with_legacy_mosaic_config_readers_and_writers(self):
        file = descriptor_pb2.FileDescriptorProto(name="legacy_mosaic.proto", syntax="proto3")
        message = file.message_type.add(name="MosaicConfig")
        ai_processor_pb2.MosaicConfig.DESCRIPTOR.CopyToProto(message)
        self.assertEqual(message.field[2].name, "mask_margin")
        self.assertEqual(message.field[2].type, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
        del message.field[2]
        del message.oneof_decl[2]
        pool = descriptor_pool.DescriptorPool()
        pool.Add(file)
        legacy_class = message_factory.GetMessageClass(pool.FindMessageTypeByName("MosaicConfig"))
        current = ai_processor_pb2.MosaicConfig(blur_radius=12, pixel_size=2, mask_margin=0)
        legacy = legacy_class.FromString(current.SerializeToString())
        self.assertEqual((legacy.blur_radius, legacy.pixel_size), (12, 2))
        legacy.DiscardUnknownFields()
        read_back = ai_processor_pb2.MosaicConfig.FromString(legacy.SerializeToString())
        self.assertFalse(read_back.HasField("mask_margin"))
        self.assertEqual((read_back.blur_radius, read_back.pixel_size), (12, 2))

    def test_additive_response_fields_are_ignored_by_a_legacy_reader(self):
        legacy_class = _legacy_processed_video_chunk_class()
        current = ai_processor_pb2.ProcessedVideoChunk(
            data=b"server-mosaic",
            mosaic_jpeg=b"deprecated-alias",
            timestamp=17,
            status_message="success",
            width=640,
            height=360,
            frame_id=19,
            processing_ms=8.5,
            error_code="new-field",
            stats=ai_processor_pb2.FrameStats(tracks=3),
        )

        legacy = legacy_class()
        legacy.ParseFromString(current.SerializeToString())

        self.assertEqual(legacy.data, b"server-mosaic")
        self.assertEqual(legacy.timestamp, 17)
        self.assertEqual(legacy.status_message, "success")
        self.assertEqual((legacy.width, legacy.height), (640, 360))
        self.assertEqual(legacy.frame_id, 19)
        self.assertEqual(legacy.processing_ms, 8.5)


def _legacy_processed_video_chunk_class():
    file_descriptor = descriptor_pb2.FileDescriptorProto(
        name="legacy_ai_processor.proto",
        syntax="proto3",
    )
    message = file_descriptor.message_type.add(name="ProcessedVideoChunk")
    fields = (
        ("data", 1, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES),
        ("timestamp", 2, descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
        ("status_message", 3, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
        ("width", 5, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
        ("height", 6, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
        ("frame_id", 7, descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
        ("processing_ms", 8, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE),
    )
    for name, number, field_type in fields:
        message.field.add(
            name=name,
            number=number,
            type=field_type,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
        )
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_descriptor)
    descriptor = pool.FindMessageTypeByName("ProcessedVideoChunk")
    return message_factory.GetMessageClass(descriptor)


if __name__ == "__main__":
    unittest.main()
