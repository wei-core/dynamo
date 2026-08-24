# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.protocols.video_protocol module."""

import pytest

from dynamo.common.protocols.video_protocol import (
    NvCreateVideoRequest,
    NvVideosResponse,
    VideoData,
    VideoNvExt,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_video_request_wire_shape():
    request = NvCreateVideoRequest(
        prompt="cat",
        model="wan",
        output_format="mp4",
        response_format="url",
        stream=True,
        nvext=VideoNvExt(boundary_ratio=0.3, guidance_scale_2=1.0),
    )

    assert request.model_dump(exclude_none=True) == {
        "prompt": "cat",
        "model": "wan",
        "response_format": "url",
        "output_format": "mp4",
        "stream": True,
        "nvext": {"boundary_ratio": 0.3, "guidance_scale_2": 1.0},
    }


def test_video_request_typed_references_preserve_wire_order():
    request = NvCreateVideoRequest(
        prompt="cat",
        model="video-model",
        input_references=[
            {"type": "image", "source": "https://example.com/cat.png"},
            {"type": "audio", "source": "data:audio/wav;base64,AA=="},
        ],
    )

    assert [reference.type for reference in request.input_references] == [
        "image",
        "audio",
    ]


@pytest.mark.parametrize(
    "input_references",
    [[], [{"type": "image", "source": "https://example.com/cat.png"}]],
)
def test_video_request_rejects_invalid_reference_combinations(input_references):
    kwargs: dict[str, object] = {"input_references": input_references}
    if input_references:
        kwargs["input_reference"] = "https://example.com/legacy.png"

    with pytest.raises(ValueError):
        NvCreateVideoRequest(prompt="cat", model="video-model", **kwargs)


def test_video_request_rejects_more_than_twelve_references():
    with pytest.raises(ValueError, match="at most 12"):
        NvCreateVideoRequest(
            prompt="cat",
            model="video-model",
            input_references=[
                {"type": "image", "source": f"https://example.com/{index}.png"}
                for index in range(13)
            ],
        )


@pytest.mark.parametrize(
    ("task", "references", "nvext", "message"),
    [
        (
            "t2va",
            [{"type": "image", "source": "https://example.com/cat.png"}],
            {},
            "does not accept",
        ),
        (
            "fl2va",
            [{"type": "audio", "source": "https://example.com/cat.wav"}],
            {},
            "only one or two image",
        ),
        (
            "fl2va",
            [{"type": "image", "source": "https://example.com/cat.png"}],
            {"frame_indices": [0, -1]},
            "one frame index per image",
        ),
        (
            "ref2va",
            [{"type": "audio", "source": "https://example.com/cat.wav"}],
            {},
            "at least one image or video",
        ),
        (
            "ref2va",
            [{"type": "video", "source": "https://example.com/cat.mp4"}],
            {"start_time_seconds": [0.0, 1.0]},
            "one value per video",
        ),
    ],
)
def test_video_request_rejects_invalid_h3_reference_contract(
    task, references, nvext, message
):
    with pytest.raises(ValueError, match=message):
        NvCreateVideoRequest(
            prompt="cat",
            model="MiniMaxAI/MiniMax-H3",
            input_references=references,
            nvext={"task": task, **nvext},
        )


@pytest.mark.parametrize(
    "references",
    [
        [{"type": "audio", "source": "https://example.com/cat.wav"}],
        [
            {"type": "image", "source": f"https://example.com/{index}.png"}
            for index in range(3)
        ],
        [
            {"type": "video", "source": f"https://example.com/{index}.mp4"}
            for index in range(4)
        ],
    ],
)
def test_video_request_rejects_invalid_taskless_h3_reference_contract(references):
    with pytest.raises(ValueError):
        NvCreateVideoRequest(
            prompt="cat",
            model="MiniMaxAI/MiniMax-H3",
            input_references=references,
        )


@pytest.mark.parametrize(
    "nvext",
    [
        {"task": "t2va", "fps": 16},
        {"task": "fl2va", "frame_indices": [1]},
        {"task": "ref2va", "duration": 3},
        {"task": "ref2va", "num_outputs_per_prompt": 11},
    ],
)
def test_video_request_rejects_invalid_h3_controls(nvext):
    with pytest.raises(ValueError):
        NvCreateVideoRequest(prompt="cat", model="MiniMaxAI/MiniMax-H3", nvext=nvext)


def test_video_response_wire_shape():
    response = NvVideosResponse(
        id="r1",
        model="wan",
        created=0,
        data=[VideoData(output_format="mp4", url="http://example.com/v.mp4")],
    )

    assert response.model_dump(exclude_none=True) == {
        "id": "r1",
        "object": "video",
        "model": "wan",
        "status": "completed",
        "progress": 100,
        "created": 0,
        "data": [{"output_format": "mp4", "url": "http://example.com/v.mp4"}],
    }


def test_video_response_reports_media_metadata():
    response = VideoData(
        output_format="mp4",
        url="http://example.com/v.mp4",
        fps=24,
        audio_sample_rate=32000,
    )

    assert response.model_dump(exclude_none=True) == {
        "output_format": "mp4",
        "url": "http://example.com/v.mp4",
        "fps": 24,
        "audio_sample_rate": 32000,
    }
