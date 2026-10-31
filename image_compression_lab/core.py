"""NumPy SVD algorithms, independent of the interface."""
from io import BytesIO
import warnings
import numpy as np
from PIL import Image, ImageOps


def prepare_image(data, max_side=384, grayscale=False):
    if len(data) > 20 * 1024 * 1024:
        raise ValueError("图片不能超过 20 MB。")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(BytesIO(data)) as source:
            if source.width * source.height > 25_000_000:
                raise ValueError("图片不能超过 2500 万像素。")
            source.seek(0)
            im = ImageOps.exif_transpose(source).convert("RGBA")
            original_size = im.size
            bg = Image.new("RGBA", im.size, "white")
            im = Image.alpha_composite(bg, im).convert("RGB")
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            if grayscale:
                im = im.convert("L")
            return np.asarray(im).copy(), original_size


def demo_image():
    y, x = np.mgrid[0:256, 0:384]
    a = np.stack([35 + 170*x/383, 70 + 140*y/255,
                  130 + 60*np.sin(x/38)*np.cos(y/32)], axis=-1)
    a[(x-275)**2 + (y-75)**2 < 36**2] = [255, 215, 110]
    a[y > 170 + 25*np.sin(x/46)] = [35, 100, 100]
    a[(x > 45) & (x < 145) & (y > 95) & (y < 220)] = [235, 125, 95]
    a[(x > 62) & (x < 128) & (y > 115) & (y < 180)] = [45, 65, 95]
    out = BytesIO()
    Image.fromarray(np.uint8(a)).save(out, format="PNG")
    return out.getvalue()


def decompose(pixels):
    """Each channel is decomposed separately; float64 for numerical accuracy."""
    a = pixels.astype(np.float64)
    if a.ndim == 2:
        a = a[:, :, None]
    return [np.linalg.svd(a[:, :, c], full_matrices=False)
            for c in range(a.shape[2])]


def reconstruct(factors, k, grayscale=False):
    if not 0 <= k <= len(factors[0][1]):
        raise ValueError("k 超出有效范围")
    channels = [(u[:, :k] * s[:k]) @ vh[:k, :] for u, s, vh in factors]
    a = np.stack(channels, axis=-1)
    return a[:, :, 0] if grayscale else a


def display_pixels(a):
    return np.rint(np.clip(a, 0, 255)).astype(np.uint8)


def metrics(reference, displayed):
    diff = reference.astype(np.float64) - displayed.astype(np.float64)
    mse = float(np.mean(diff**2))
    psnr = float("inf") if mse == 0 else float(10*np.log10(255**2/mse))
    return mse, psnr, float(np.mean(np.abs(diff)))


def storage(shape, k, bytes_per_value=4):
    h, w = shape[:2]
    channels = 1 if len(shape) == 2 else shape[2]
    original = h*w*channels  # uint8 pixels
    count = channels*k*(h+w+1)
    return original, count, count*bytes_per_value


def theoretical_curve(factors, shape, bytes_per_value=4):
    energy = np.sum([s**2 for _, s, _ in factors], axis=0)
    remaining = np.r_[np.cumsum(energy[::-1])[::-1], 0.0]
    total = float(remaining[0])
    result = []
    for k, tail in enumerate(remaining):
        original, count, size = storage(shape, k, bytes_per_value)
        result.append({"k": k, "理论MSE": float(tail/np.prod(shape)),
                       "保留能量(%)": 100.0 if total == 0 else float(100*(1-tail/total)),
                       "因子存储字节": size, "占原始像素(%)": 100*size/original})
    return result


def png_bytes(pixels):
    out = BytesIO()
    Image.fromarray(pixels).save(out, format="PNG")
    return out.getvalue()


def factor_bytes(factors, k, shape, dtype="float32"):
    payload = {"shape": np.array(shape), "k": np.array(k)}
    for c, (u, s, vh) in enumerate(factors):
        payload.update({f"U_{c}": u[:, :k].astype(dtype),
                        f"s_{c}": s[:k].astype(dtype),
                        f"Vh_{c}": vh[:k, :].astype(dtype)})
    out = BytesIO()
    np.savez(out, **payload)
    return out.getvalue()
