import rtms from "@zoom/rtms";
import { WebSocket } from "ws";

const CHUNK_SIZE_BYTES = 1280;
const SEND_INTERVAL_MS = 40;

let clients = new Map();

export function startRtmsHandler(config) {
  rtms.onWebhookEvent(({ event, payload }) => {
    const streamId = payload?.rtms_stream_id;

    if (event == "meeting.rtms_stopped") {
      if (!streamId) {
        console.log(`Received meeting.rtms_stopped event without stream ID`);
        return;
      }
      const session = clients.get(streamId);
      if (!session) {
        console.log(
          `Received meeting.rtms_stopped event for unknown stream ID: ${streamId}`,
        );
        return;
      }
      if (session.sendInterval) {
        clearInterval(session.sendInterval);
      }
      session.zoomClient.leave();
      if (session.wsClient) {
        session.wsClient.close();
        console.log(`Closed WebSocket connection for stream: ${streamId}`);
      }
      clients.delete(streamId);
      return;
    } else if (event !== "meeting.rtms_started") {
      console.log(`Ignoring unknown event: ${event}`);
      return;
    }

    console.log(`RTMS stream started for stream ID: ${streamId}`);

    const wsClient = new WebSocket(config.transcriptionServer);
    let audioBuffer = Buffer.alloc(0);
    let sendInterval = null;

    wsClient.on("open", () => {
      console.log(`WebSocket connection opened for stream: ${streamId}`);
      sendInterval = setInterval(() => {
        if (audioBuffer.length >= CHUNK_SIZE_BYTES) {
          const chunk = audioBuffer.slice(0, CHUNK_SIZE_BYTES);
          audioBuffer = audioBuffer.slice(CHUNK_SIZE_BYTES);

          const session = clients.get(streamId);
          const currentUserName = session?.userName || "Zoom User";

          const payload = {
            userName: currentUserName,
            audio: chunk.toString("base64"),
          };

          if (wsClient.readyState === WebSocket.OPEN) {
            wsClient.send(JSON.stringify(payload));
          }
        }
      }, SEND_INTERVAL_MS);
    });

    wsClient.on("error", (error) => {
      console.error(`WebSocket error for stream ${streamId}:`, error);
    });

    wsClient.on("close", () => {
      console.log(`WebSocket connection closed for stream: ${streamId}`);
      if (sendInterval) {
        clearInterval(sendInterval);
      }
    });

    const zoomClient = new rtms.Client();
    clients.set(streamId, {
      zoomClient,
      wsClient,
      sendInterval,
      userName: "Zoom User",
    });

    zoomClient.onTranscriptData((data, size, timestamp, metadata) => {
      console.log(`[${timestamp}] -- ${metadata.userName}: ${data}`);
    });

    zoomClient.onAudioData((data, size, timestamp, metadata) => {
      const session = clients.get(streamId);
      if (session) {
        session.userName = metadata.userName;
      }
      audioBuffer = Buffer.concat([audioBuffer, data]);
    });

    const video_params = {
      contentType: rtms.VideoContentType.RAW_VIDEO,
      codec: rtms.VideoCodec.H264,
      resolution: rtms.VideoResolution.SD,
      dataOpt: rtms.VideoDataOption.VIDEO_SINGLE_ACTIVE_STREAM,
      fps: 30,
    };

    zoomClient.setVideoParams(video_params);
    zoomClient.onVideoData((data, size, timestamp, metadata) => {
      console.log(
        `Received ${size} bytes of video data at ${timestamp} from ${metadata.userName}`,
      );
    });

    zoomClient.setDeskshareParams(video_params);
    zoomClient.onDeskshareData((data, size, timestamp, metadata) => {
      console.log(
        `Received ${size} bytes of deskshare data at ${timestamp} from ${metadata.userName}`,
      );
    });

    zoomClient.join(payload);
  });

  console.log("RTMS webhook handler initialized.");
}
