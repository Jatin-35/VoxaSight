# Old version code v0.x

import asyncio
from typing import Annotated

from livekit.agents import JobContext, WorkerOptions, run_app
from livekit.agents import llm, tokenize, tts, voice_assistant
from livekit.agents.llm import ChatContext, ChatImage, ChatMessage
from livekit.plugins import deepgram, openai, silero
from livekit.rtc import Room, RemoteVideoTrack, VideoFrame

class AssistantFunction(llm.FunctionContext):
    """This class defines functions called by the assistant."""

    @llm.ai_callable(
        description=(
            "Called when asked to evaluate something that would require vision capabilities,"
            "for example, an image, video, or the webcam feed."
        )
    )
    async def image(
        self,
        user_msg: Annotated[
            str,
            llm.TypeInfo(
                description="The user message that triggered this function"
            ),
        ],
    ):
        print(f"Message triggering vision capabilities: {user_msg}")
        return None

async def get_video_track(room: Room):
    """Get the first video track from the room to process images."""
    video_track = asyncio.Future[RemoteVideoTrack]()

    for _, participant in room.remote_participants.items():
        for _, track_publication in participant.track_publications.items():
            if track_publication.track is not None and isinstance(
                track_publication.track, RemoteVideoTrack
            ):
                video_track.set_result(track_publication.track)
                print(f"Using video track {track_publication.track.sid}")
                break

    return await video_track

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"Room name: {ctx.room.name}")

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Your name is Alloy. You are a funny, witty bot. Your interface with users will be voice and vision."
                    "Respond with short and concise answers. Avoid using unpronouncable punctuation or emojis."
                ),
            )
        ]
    )

    gpt = openai.LLM(model="gpt-4o")

    # Use StreamAdapter for OpenAI TTS to make it compatible with VoiceAssistant
    openai_tts = tts.StreamAdapter(
        tts=openai.TTS(voice="alloy"),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
    )

    latest_image: VideoFrame | None = None

    assistant = voice_assistant.VoiceAssistant(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=gpt,
        tts=openai_tts,
        fnc_ctx=AssistantFunction(),
        chat_ctx=chat_context,
    )

    chat = ctx.room.chat

    async def _answer(text: str, use_image: bool = False):
        """Answer the user's message with text and optionally the latest image."""
        content: list[str | ChatImage] = [text]
        if use_image and latest_image:
            content.append(ChatImage(image=latest_image))

        chat_context.messages.append(ChatMessage(role="user", content=content))

        stream = gpt.chat(chat_ctx=chat_context)
        await assistant.say(stream, allow_interruptions=True)

    @chat.on("message_received")
    def on_message_received(msg):
        """Handle new messages from the user."""
        if msg.message:
            asyncio.create_task(_answer(msg.message, use_image=False))

    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[llm.CalledFunction]):
        """Handle completion of assistant's function calls."""
        if len(called_functions) == 0:
            return

        user_msg = called_functions[0].call_info.arguments.get("user_msg")
        if user_msg:
            asyncio.create_task(_answer(user_msg, use_image=True))

    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say("Hi there! How can I help?", allow_interruptions=True)

    while ctx.room.connection_state == "connected":
        video_track = await get_video_track(ctx.room)

        async for event in video_track.stream():
            # Continually grab the latest image from the video track
            latest_image = event.frame

if __name__ == "__main__":
    run_app(WorkerOptions(entrypoint=entrypoint))