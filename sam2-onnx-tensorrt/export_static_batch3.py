import os
import torch
import onnx
from src.Module import ImageEncoder
from sam2.build_sam import build_sam2

def export_image_encoder(model, onnx_path):
    print(">>> Exporting Image Encoder (Batch 3)...")
    input_img = torch.randn(3, 3, 1024, 1024).cpu()
    out = model(input_img)
    output_names = ["pix_feat", "high_res_feat0", "high_res_feat1", "vision_feats", "vision_pos_embed"]
    
    torch.onnx.export(
        model,
        input_img,
        os.path.join(onnx_path, "image_encoder.onnx"),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=output_names,
    )
    onnx_model = onnx.load(os.path.join(onnx_path, "image_encoder.onnx"))
    onnx.checker.check_model(onnx_model)
    print("[SUCCESS] Image Encoder (Batch 3) exported successfully!")

if __name__ == "__main__":
    outdir = "/home/jianwei/vsim/unitree_g1_hw_endpoint/image_processing/sam2_onnx/tiny_batch3/"
    os.makedirs(outdir, exist_ok=True)
    
    config = "configs/sam2.1/sam2.1_hiera_t.yaml"
    checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
    
    sam2_model = build_sam2(config, checkpoint, device="cpu")
    image_encoder = ImageEncoder(sam2_model).cpu()
    
    export_image_encoder(image_encoder, outdir)
