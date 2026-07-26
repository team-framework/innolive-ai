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

        video = ai_processor_pb2.VideoChunk.DESCRIPTOR
        processed = ai_processor_pb2.ProcessedVideoChunk.DESCRIPTOR
        self.assertTrue(video.fields_by_name["batch_size"].GetOptions().deprecated)
        self.assertTrue(processed.fields_by_name["data"].GetOptions().deprecated)

    def test_additive_response_fields_are_ignored_by_a_legacy_reader(self):
        legacy_class = _legacy_processed_video_chunk_class()
        current = ai_processor_pb2.ProcessedVideoChunk(
            data=b"legacy-pixel-field",
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

        self.assertEqual(legacy.data, b"legacy-pixel-field")
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
