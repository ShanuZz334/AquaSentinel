"""
AquaSentinel - Core Computer Vision Module
Physics-Based Optical Restoration, Spectral Analytics & Density Heatmap Engine

Implements multi-stage underwater optical restoration:
1. Beer-Lambert Red-Wavelength Absorption Compensation
2. Underwater Dark Channel Prior (UDCP) Backscattering Suppression
3. Color Constancy White-Balancing (Modified Gray World)
4. Spectral RGB Histogram & Channel Telemetry Extraction
5. Spatial Hazard Density Heatmap Generation
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple, Dict, Any, List


class UnderwaterOpticalEnhancer:
    def __init__(self, red_gain: float = 1.35, blue_damping: float = 0.92, contrast_factor: float = 1.25):
        self.red_gain = red_gain
        self.blue_damping = blue_damping
        self.contrast_factor = contrast_factor

    def compensate_red_attenuation(self, img_np: np.ndarray) -> np.ndarray:
        """
        Inverts differential light attenuation according to Beer-Lambert law:
        I(d, lambda) = I_0 * exp(-beta(lambda) * d)
        """
        img_float = img_np.astype(np.float32) / 255.0
        r, g, b = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]

        mean_r = np.mean(r)
        mean_g = np.mean(g)

        # Dynamic red compensation based on green-to-red deficiency
        diff = max(0.0, float(mean_g - mean_r))
        r_compensated = r + (self.red_gain * diff * (1.0 - r) * g)
        b_damped = b * self.blue_damping

        enhanced = np.stack([r_compensated, g, b_damped], axis=-1)
        return np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)

    def underwater_dark_channel_prior(self, img_np: np.ndarray, patch_size: int = 15) -> np.ndarray:
        """
        UDCP: Utilizes Green and Blue channels to estimate the transmission map
        since the Red channel is severely attenuated in aquatic medium.
        """
        img_float = img_np.astype(np.float32) / 255.0
        # UDCP uses min(G, B) rather than min(R, G, B)
        dark_channel = np.minimum(img_float[:, :, 1], img_float[:, :, 2])
        
        # Estimate atmospheric/background water light (top 0.1% brightest in dark channel)
        num_pixels = dark_channel.size
        num_top = max(int(num_pixels * 0.001), 1)
        flat_dark = dark_channel.flatten()
        indices = np.argpartition(flat_dark, -num_top)[-num_top:]
        water_light = np.mean(img_float.reshape(-1, 3)[indices], axis=0)
        water_light = np.maximum(water_light, 0.2)

        # Transmission map estimation
        omega = 0.85
        norm_img = img_float / water_light
        transmission = 1.0 - (omega * np.minimum(norm_img[:, :, 1], norm_img[:, :, 2]))
        transmission = np.clip(transmission, 0.2, 1.0)[:, :, np.newaxis]

        # Dehazed restoration
        recovered = ((img_float - water_light) / transmission) + water_light
        return np.clip(recovered * 255.0, 0, 255).astype(np.uint8)

    def adaptive_color_balance(self, img_np: np.ndarray, clip_percent: float = 1.0) -> np.ndarray:
        """
        White balance color constancy to eliminate pervasive cyan/green cast.
        """
        out = np.zeros_like(img_np)
        for i in range(3):
            channel = img_np[:, :, i]
            low_val = np.percentile(channel, clip_percent)
            high_val = np.percentile(channel, 100.0 - clip_percent)

            if high_val > low_val:
                scaled = (channel.astype(np.float32) - low_val) * (255.0 / (high_val - low_val))
                out[:, :, i] = np.clip(scaled, 0, 255).astype(np.uint8)
            else:
                out[:, :, i] = channel
        return out

    def compute_spectral_histogram(self, img_np: np.ndarray, bins: int = 16) -> Dict[str, List[float]]:
        """
        Computes 16-bin normalized histograms for R, G, and B spectral bands.
        """
        hist_data = {}
        for idx, color in enumerate(["R", "G", "B"]):
            hist, _ = np.histogram(img_np[:, :, idx], bins=bins, range=(0, 256), density=True)
            hist_data[color] = [round(float(v) * 100, 2) for v in hist]
        return hist_data

    def generate_density_heatmap(self, image: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
        """
        Renders a continuous spatial Gaussian density heatmap over high-threat coordinates.
        """
        width, height = image.size
        density_map = np.zeros((height, width), dtype=np.float32)

        for det in detections:
            ymin, xmin, ymax, xmax = det["bbox_norm"]
            cx = int(((xmin + xmax) / 2.0) * width)
            cy = int(((ymin + ymax) / 2.0) * height)
            w_box = max(int((xmax - xmin) * width), 20)
            h_box = max(int((ymax - ymin) * height), 20)
            sigma = max(w_box, h_box) / 2.0

            # Threat weight
            threat_weight = 1.0
            if det.get("hazard_level") == "CRITICAL":
                threat_weight = 2.5
            elif det.get("hazard_level") in ["WARNING", "MODERATE"]:
                threat_weight = 1.5

            # Apply 2D Gaussian kernel on sub-grid
            y_coords, x_coords = np.ogrid[:height, :width]
            dist_sq = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
            kernel = np.exp(-dist_sq / (2.0 * (sigma ** 2))) * threat_weight
            density_map += kernel

        # Normalize density map to [0, 255]
        max_val = np.max(density_map)
        if max_val > 0:
            density_norm = (density_map / max_val * 255.0).astype(np.uint8)
        else:
            density_norm = np.zeros((height, width), dtype=np.uint8)

        # Colorize density map (Blue -> Cyan -> Yellow -> Red)
        heatmap_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        norm_float = density_norm.astype(np.float32) / 255.0

        heatmap_rgba[:, :, 0] = np.clip(norm_float * 2.0 * 255.0, 0, 255).astype(np.uint8)  # Red
        heatmap_rgba[:, :, 1] = np.clip((1.0 - np.abs(norm_float - 0.5) * 2.0) * 255.0, 0, 255).astype(np.uint8)  # Green
        heatmap_rgba[:, :, 2] = np.clip((1.0 - norm_float) * 255.0, 0, 255).astype(np.uint8)  # Blue
        heatmap_rgba[:, :, 3] = np.clip(norm_float * 180.0, 0, 180).astype(np.uint8)  # Alpha

        heatmap_img = Image.fromarray(heatmap_rgba, mode="RGBA")
        base_img = image.convert("RGBA")
        blended = Image.alpha_composite(base_img, heatmap_img)
        return blended.convert("RGB")

    def enhance(self, pil_image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Executes full physical restoration pipeline and extracts spectral telemetry.
        """
        original_np = np.array(pil_image.convert("RGB"))

        # Step 1: Red channel attenuation recovery
        compensated_np = self.compensate_red_attenuation(original_np)

        # Step 2: UDCP backscatter suppression
        dehazed_np = self.underwater_dark_channel_prior(compensated_np)

        # Step 3: Color constancy balance
        balanced_np = self.adaptive_color_balance(dehazed_np)

        # Step 4: Fine contrast and edge unsharp masking
        enhanced_pil = Image.fromarray(balanced_np)
        contrast_enhancer = ImageEnhance.Contrast(enhanced_pil)
        enhanced_pil = contrast_enhancer.enhance(self.contrast_factor)
        enhanced_pil = enhanced_pil.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))

        enhanced_np = np.array(enhanced_pil)

        # Spectral Metrics Calculation
        orig_r_mean = float(np.mean(original_np[:, :, 0]))
        orig_g_mean = float(np.mean(original_np[:, :, 1]))
        orig_b_mean = float(np.mean(original_np[:, :, 2]))

        enh_r_mean = float(np.mean(enhanced_np[:, :, 0]))
        enh_g_mean = float(np.mean(enhanced_np[:, :, 1]))
        enh_b_mean = float(np.mean(enhanced_np[:, :, 2]))

        orig_cast_ratio = orig_g_mean / (orig_r_mean + 1e-5)
        enh_cast_ratio = enh_g_mean / (enh_r_mean + 1e-5)
        reduction_pct = max(0.0, min(100.0, (1.0 - (enh_cast_ratio / orig_cast_ratio)) * 100))

        telemetry = {
            "original_rgb_means": {"R": round(orig_r_mean, 1), "G": round(orig_g_mean, 1), "B": round(orig_b_mean, 1)},
            "enhanced_rgb_means": {"R": round(enh_r_mean, 1), "G": round(enh_g_mean, 1), "B": round(enh_b_mean, 1)},
            "optical_cast_reduction_pct": round(reduction_pct, 1),
            "red_attenuation_recovery_factor": round(enh_r_mean / (orig_r_mean + 1e-5), 2),
            "udcp_applied": True,
            "raw_histograms": self.compute_spectral_histogram(original_np),
            "enhanced_histograms": self.compute_spectral_histogram(enhanced_np)
        }

        return enhanced_pil, telemetry
