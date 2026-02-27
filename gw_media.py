import io
import time

from mss import mss
from PIL import Image

import gw_state as st


def capture_screenshot_to_s3_url(
    max_side_px: int = 960, jpeg_quality: int = 62, expires: int = 90
) -> tuple[str | None, str | None]:
    if not st.SCREENSHOT_ENABLED:
        return None, None
    if not st.s3 or not st.S3_BUCKET:
        return None, None
    try:
        with mss() as sct:
            raw = sct.grab(sct.monitors[1])
            img = Image.frombytes("RGB", raw.size, raw.rgb)
        w, h = img.size
        scale = max(w, h) / float(max_side_px)
        if scale > 1:
            img = img.resize((int(w / scale), int(h / scale)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        buf.seek(0)

        ts = int(time.time() * 1000)
        key = f"{st.S3_PREFIX}shot_{ts}.jpg"
        st.s3.upload_fileobj(
            buf,
            st.S3_BUCKET,
            key,
            ExtraArgs={"ContentType": "image/jpeg", "ACL": "private"},
        )
        url = st.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": st.S3_BUCKET, "Key": key},
            ExpiresIn=expires,
        )
        global_host = ".s3.amazonaws.com"
        regional_host = f".s3.{st.AWS_REGION}.amazonaws.com"
        if global_host in url and regional_host not in url:
            url = url.replace(global_host, regional_host)
        return url, key
    except Exception as e:
        print(f"[Screenshot/S3] Failed: {e}")
        return None, None


def delete_s3_object(key: str):
    if not st.s3 or not st.S3_BUCKET:
        return
    try:
        st.s3.delete_object(Bucket=st.S3_BUCKET, Key=key)
    except Exception as e:
        print(f"[Screenshot/S3] Delete failed: {e}")

