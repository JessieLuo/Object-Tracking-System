from streaming.subscribe import FrameSource
from streaming.vid_pub import VideoFileWriter


def main():
    src = "streaming/data/test_640.mp4"
    source = FrameSource(src)

    writer = VideoFileWriter("out.mp4", (640, 480), 30)

    opened = False

    try:
        for frame in source.frames():
            if not opened:
                h, w = frame.shape[:2]
                writer.size_wh = (w, h)
                writer.open()
                opened = True

            writer.write(frame)

    except KeyboardInterrupt:
        print("Interrupted, closing writer...")

    finally:
        if opened:
            writer.close()
            print("writer closed")


if __name__ == "__main__":
    main()