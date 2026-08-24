# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol types for video generation.

These types match the Rust protocol types in lib/llm/src/protocols/openai/videos.rs
to ensure compatibility with the Dynamo HTTP frontend.
"""
# TODO: Replace these Pydantic models with Python bindings to the Rust protocol types once PyO3 bindings are available.

from typing import Literal, Optional, Union

from pydantic import BaseModel, model_validator

H3Task = Literal["t2va", "fl2va", "ref2va"]


def is_minimax_h3_model_name(model: str) -> bool:
    """Return whether a public model identifier names MiniMax-H3."""
    return "minimax-h3" in model.lower().replace("_", "-")


class VideoInputReference(BaseModel):
    """Typed conditioning input for video generation."""

    type: Literal["image", "video", "audio"]
    """Reference media type."""

    source: str
    """HTTP(S), data, file URL, or an allowed local path."""


class VideoNvExt(BaseModel):
    """NVIDIA extensions for video generation requests.

    Matches Rust NvExt in lib/llm/src/protocols/openai/videos/nvext.rs.
    """

    annotations: Optional[list[str]] = None
    """Annotations for SSE stream events."""

    fps: Optional[int] = None
    """Frames per second (default: 24)."""

    num_frames: Optional[int] = None
    """Number of frames to generate (overrides fps * seconds if set)."""

    negative_prompt: Optional[str] = None
    """Optional negative prompt."""

    num_inference_steps: Optional[int] = None
    """Number of denoising steps (default: 50)."""

    guidance_scale: Optional[float] = None
    """CFG guidance scale (default: 5.0)."""

    seed: Optional[int] = None
    """Random seed for reproducibility."""

    boundary_ratio: Optional[float] = None
    """MoE expert switching boundary as a fraction of the denoising schedule (vLLM-Omni I2V)."""

    guidance_scale_2: Optional[float] = None
    """CFG scale for the low-noise expert (vLLM-Omni I2V dual-guidance)."""

    task: Optional[H3Task] = None
    """MiniMax-H3 task routed to its FL2VA or Ref2VA transformer."""

    duration: Optional[float] = None
    """Requested MiniMax-H3 duration in seconds (4 through 15)."""

    flow_shift: Optional[float] = None
    """MiniMax-H3 video sigma shift."""

    audio_flow_shift: Optional[float] = None
    """MiniMax-H3 audio sigma shift."""

    aspect_ratio: Optional[str] = None
    """MiniMax-H3 output aspect ratio."""

    short_edge: Optional[int] = None
    """MiniMax-H3 output canvas short edge."""

    frame_indices: Optional[list[int]] = None
    """FL2VA keyframe positions: [0], [-1], or [0, -1]."""

    start_time_seconds: Optional[Union[float, list[float]]] = None
    """Start offset for one reference video, or one offset per video."""

    num_outputs_per_prompt: Optional[int] = None
    """Number of generated videos (MiniMax-H3 supports 1 through 10)."""

    quality: Optional[Literal["lossless", "high"]] = None
    """MiniMax-H3 request-scoped quality policy."""

    @model_validator(mode="after")
    def validate_h3_fields(self) -> "VideoNvExt":
        if self.duration is not None and not 4 <= self.duration <= 15:
            raise ValueError("duration must be between 4 and 15 seconds")
        if self.num_outputs_per_prompt is not None and not (
            1 <= self.num_outputs_per_prompt <= 10
        ):
            raise ValueError("num_outputs_per_prompt must be between 1 and 10")
        if self.task is not None and self.fps is not None and self.fps != 24:
            raise ValueError("MiniMax-H3 fps is fixed at 24")
        if self.task == "fl2va" and self.frame_indices is not None:
            if self.frame_indices not in ([0], [-1], [0, -1]):
                raise ValueError("FL2VA frame_indices must be [0], [-1], or [0, -1]")
        return self


class NvCreateVideoRequest(BaseModel):
    """Request for video generation (/v1/videos endpoint).

    Matches Rust NvCreateVideoRequest in lib/llm/src/protocols/openai/videos.rs.
    """

    # Required fields
    prompt: str
    """The text prompt for video generation."""

    model: str
    """The model to use for video generation."""

    # Optional fields
    input_reference: Optional[str] = None
    """Optional image reference that guides generation (for I2V)."""

    input_references: Optional[list[VideoInputReference]] = None
    """Typed references; order is preserved within each media type."""

    seconds: Optional[int] = None
    """Clip duration in seconds."""

    size: Optional[str] = None
    """Video size in WxH format (default: '832x480')."""

    user: Optional[str] = None
    """Optional user identifier."""

    response_format: Optional[Literal["url", "b64_json"]] = None
    """How the generated data should be returned: 'url' or 'b64_json'.
    If unset, handlers default to 'url'."""

    output_format: Optional[str] = None
    """Requested container format (e.g. 'mp4', 'mjpeg').
    This is a hint; check output_format in the response data for the actual format."""

    stream: Optional[bool] = None
    """Whether to stream the video generation (default: false)."""

    nvext: Optional[VideoNvExt] = None
    """NVIDIA extensions."""

    @model_validator(mode="after")
    def validate_input_references(self) -> "NvCreateVideoRequest":
        if self.input_reference is not None and self.input_references is not None:
            raise ValueError(
                "input_reference and input_references are mutually exclusive"
            )
        if self.input_references is not None and not self.input_references:
            raise ValueError("input_references must not be empty")
        if self.input_references is not None and len(self.input_references) > 12:
            raise ValueError("input_references accepts at most 12 references")

        task = self.nvext.task if self.nvext is not None else None
        if task is not None or is_minimax_h3_model_name(self.model):
            self.validate_h3_reference_contract(task or self.infer_h3_task())
        return self

    def infer_h3_task(self) -> H3Task:
        """Match MiniMax-H3's task inference from the supplied references."""
        if self.nvext is not None and self.nvext.task is not None:
            return self.nvext.task
        reference_types = {reference.type for reference in self.input_references or []}
        if reference_types.intersection({"video", "audio"}):
            return "ref2va"
        if self.input_reference is not None or "image" in reference_types:
            return "fl2va"
        return "t2va"

    def validate_h3_reference_contract(self, task: H3Task) -> None:
        """Validate controls and references for a resolved MiniMax-H3 task."""
        nvext = self.nvext or VideoNvExt()
        if nvext.fps is not None and nvext.fps != 24:
            raise ValueError("MiniMax-H3 fps is fixed at 24")
        if task == "fl2va" and nvext.frame_indices is not None:
            if nvext.frame_indices not in ([0], [-1], [0, -1]):
                raise ValueError("FL2VA frame_indices must be [0], [-1], or [0, -1]")

        counts = {kind: 0 for kind in ("image", "video", "audio")}
        if self.input_reference is not None:
            counts["image"] = 1
        for reference in self.input_references or []:
            counts[reference.type] += 1
        total = sum(counts.values())

        if task == "t2va" and total:
            raise ValueError("t2va does not accept input references")
        if task == "fl2va":
            if counts["video"] or counts["audio"] or not counts["image"]:
                raise ValueError("fl2va accepts only one or two image references")
            if counts["image"] > 2:
                raise ValueError("fl2va accepts at most two image references")
            if (
                nvext.frame_indices is not None
                and len(nvext.frame_indices) != counts["image"]
            ):
                raise ValueError("fl2va requires one frame index per image reference")
        if task == "ref2va":
            if not counts["image"] and not counts["video"]:
                raise ValueError(
                    "ref2va requires at least one image or video reference"
                )
            if counts["image"] > 9:
                raise ValueError("ref2va accepts at most 9 image references")
            if counts["video"] > 3:
                raise ValueError("ref2va accepts at most 3 video references")
            if counts["audio"] > 3:
                raise ValueError("ref2va accepts at most 3 audio references")

        start_times = nvext.start_time_seconds
        if isinstance(start_times, list) and len(start_times) != counts["video"]:
            raise ValueError(
                "start_time_seconds requires one value per video reference"
            )
        if isinstance(start_times, float) and counts["video"] != 1:
            raise ValueError(
                "scalar start_time_seconds requires exactly one video reference"
            )


class VideoData(BaseModel):
    """Video data in response.

    Matches Rust VideoData in lib/llm/src/protocols/openai/videos.rs.
    """

    output_format: str
    """Actual container format of this video."""

    url: Optional[str] = None
    """URL of the generated video (if response_format is 'url')."""

    b64_json: Optional[str] = None
    """Base64-encoded video (if response_format is 'b64_json')."""

    fps: Optional[int] = None
    """Actual video frame rate when reported by the model."""

    audio_sample_rate: Optional[int] = None
    """Muxed audio sample rate when the generated video contains audio."""


class NvVideosResponse(BaseModel):
    """Response structure for video generation.

    Matches Rust NvVideosResponse in lib/llm/src/protocols/openai/videos.rs.
    """

    id: str
    """Unique identifier for the response."""

    object: str = "video"
    """Object type (always 'video')."""

    model: str
    """Model used for generation."""

    status: str = "completed"
    """Generation status."""

    progress: int = 100
    """Progress percentage (0-100)."""

    created: int
    """Unix timestamp of creation."""

    data: list[VideoData] = []
    """List of generated videos."""

    error: Optional[str] = None
    """Error message if generation failed."""

    inference_time_s: Optional[float] = None
    """Inference time in seconds."""
