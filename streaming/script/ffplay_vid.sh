ffplay \
-fflags nobuffer \
-flags low_delay \
-framedrop \
-strict experimental \
-analyzeduration 0 \
-probesize 32 \
-sync ext \
rtsp://127.0.0.1:8554/out