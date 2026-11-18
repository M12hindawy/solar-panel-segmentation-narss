import os, sys, math, json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

class SegFormerB4(nn.Module):
    def __init__(self, num_classes=1, embedding_dim=256):
        super().__init__()
        self.patch_embed1 = nn.Sequential(nn.Conv2d(3, 64, 7, stride=4, padding=3), nn.BatchNorm2d(64), nn.GELU())
        self.patch_embed2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU())
        self.patch_embed3 = nn.Sequential(nn.Conv2d(128, 320, 3, stride=2, padding=1), nn.BatchNorm2d(320), nn.GELU())
        self.patch_embed4 = nn.Sequential(nn.Conv2d(320, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.GELU())

        self.linear_c4 = nn.Sequential(nn.Conv2d(512, embedding_dim, 1), nn.BatchNorm2d(embedding_dim), nn.ReLU())
        self.linear_c3 = nn.Sequential(nn.Conv2d(320, embedding_dim, 1), nn.BatchNorm2d(embedding_dim), nn.ReLU())
        self.linear_c2 = nn.Sequential(nn.Conv2d(128, embedding_dim, 1), nn.BatchNorm2d(embedding_dim), nn.ReLU())
        self.linear_c1 = nn.Sequential(nn.Conv2d(64, embedding_dim, 1), nn.BatchNorm2d(embedding_dim), nn.ReLU())

        self.fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, 1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(),
            nn.Conv2d(embedding_dim, num_classes, 1)
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        c1 = self.patch_embed1(x)
        c2 = self.patch_embed2(c1)
        c3 = self.patch_embed3(c2)
        c4 = self.patch_embed4(c3)

        c1_p = self.linear_c1(c1)
        c2_p = F.interpolate(self.linear_c2(c2), size=c1_p.shape[2:], mode='bilinear', align_corners=False)
        c3_p = F.interpolate(self.linear_c3(c3), size=c1_p.shape[2:], mode='bilinear', align_corners=False)
        c4_p = F.interpolate(self.linear_c4(c4), size=c1_p.shape[2:], mode='bilinear', align_corners=False)

        fused = self.fuse(torch.cat([c1_p, c2_p, c3_p, c4_p], dim=1))
        return F.interpolate(fused, size=(h, w), mode='bilinear', align_corners=False)

def run_inference(image_path: str, checkpoint_path: str, output_geojson: str, gsd_nominal: float = 0.30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegFormerB4().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    raw_img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = raw_img.size
    img_resized = raw_img.resize((256, 256), Image.BILINEAR)

    img_np = (np.array(img_resized, dtype=np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_t = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred_logits = model(img_t)
        pred_prob = torch.sigmoid(pred_logits)[0, 0].cpu().numpy()

    binary_mask = (pred_prob > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []
    for c_idx, contour in enumerate(contours):
        if cv2.contourArea(contour) < 5: continue
        poly_coords = []
        for pt in contour[:, 0]:
            px = float(pt[0] / 256.0 * orig_w)
            py = float(pt[1] / 256.0 * orig_h)
            poly_coords.append([round(px, 2), round(py, 2)])

        if len(poly_coords) >= 3:
            poly_coords.append(poly_coords[0])
            area_m2 = cv2.contourArea(contour) * (gsd_nominal ** 2) * ((orig_w / 256.0) ** 2)
            features.append({
                "type": "Feature",
                "properties": {
                    "panel_id": c_idx,
                    "confidence": round(float(pred_prob[binary_mask > 0].mean()), 3),
                    "area_m2": round(float(area_m2), 2),
                    "nominal_gsd_m": gsd_nominal,
                    "crs_note": "Local planar coordinate projection"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [poly_coords]
                }
            })

    geojson_out = {"type": "FeatureCollection", "features": features}
    with open(output_geojson, 'w', encoding='utf-8') as f:
        json.dump(geojson_out, f, indent=2)
    print(f"[OK] Inference complete: {len(features)} panels extracted to {output_geojson}")

if __name__ == '__main__':
    print("Inference Pipeline Module Ready.")
