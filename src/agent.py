# VoxaSight v1.x – A Voice and Vision AI Assistant
# Developed using LiveKit Agents 1.x, Deepgram STT & TTS, Google Gemini LLM, Silero VAD, and Cartesia TTS.

from dotenv import load_dotenv
import asyncio
from livekit import agents, rtc
from livekit.agents import cli
from livekit.plugins.openai import LLM  # Optional import if using OpenAI
from livekit.agents import (
    Agent,
    AgentSession,
    RoomInputOptions,
    get_job_context,
    JobContext,
    ChatContext,
    ChatMessage,
)
from livekit.agents.llm import ImageContent
from livekit.plugins import (

    noise_cancellation,
    deepgram,
    silero,
    google,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Load environment variables (e.g., API keys)
load_dotenv()

class Assistant(Agent):
    """
    Custom Voice + Vision Agent extending LiveKit's Agent class.
    Handles video frames and injects them as ImageContent for LLM processing.
    """
    def __init__(self) -> None:
        self._latest_frame = None
        self._video_stream = None
        self._tasks = []
        super().__init__(
            instructions="You are a helpful voice AI assistant that can see and hear the user."
        )

    async def on_enter(self):
        """
        Called when the agent joins the room.
        Sets up video track subscription and processing.
        """
        room = get_job_context().room

        # Get remote participant and attempt to access their video track
        remote_participant = list(room.remote_participants.values())[0]
        video_tracks = [
            publication.track
            for publication in list(remote_participant.track_publications.values())
            if publication.track.kind == rtc.TrackKind.KIND_VIDEO
        ]
        if video_tracks:
            self._create_video_stream(video_tracks[0])

        # Subscribe to new video tracks dynamically
        @room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                self._create_video_stream(track)

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        """
        Called after user's voice input is processed.
        Adds the latest video frame as ImageContent to enable vision in LLM response.
        """
        if self._latest_frame:
            new_message.content.append(ImageContent(image=self._latest_frame))

    def _create_video_stream(self, track: rtc.Track):
        """
        Creates and manages a video stream reader that continually updates the latest frame.
        """
        # Close any existing stream
        if self._video_stream is not None:
            self._video_stream.close()

        self._video_stream = rtc.VideoStream(track)

        async def read_stream():
            async for event in self._video_stream:
                self._latest_frame = event.frame  # Store the latest frame for LLM

        # Launch background task to process video frames
        task = asyncio.create_task(read_stream())
        task.add_done_callback(lambda t: self._tasks.remove(t))
        self._tasks.append(task)


async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for the VoxaSight AI agent.
    Sets up session with STT, LLM, TTS, VAD, Turn Detection and Room Configurations.
    """
    await ctx.connect()

    # Define the agent session with STT, LLM, TTS, VAD, and Turn Detection
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),  # Speech to text
        llm=google.LLM(  # Language model (Google Gemini)
            model="gemini-2.0-flash-exp",
            temperature=0.8,
        ),
        tts=deepgram.TTS(model="aura-asteria-en"),  # Text to speech
        vad=silero.VAD.load(),  # Voice Activity Detection
        turn_detection=MultilingualModel(),  # Detect turn-taking in multilingual scenarios
    )

    # Start session and run the assistant
    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),  # Background Voice Cancellation
        ),
    )

    # Send initial greeting message
    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )

# CLI Entrypoint
if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
