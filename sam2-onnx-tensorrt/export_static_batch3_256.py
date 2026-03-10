import os
import torch
import onnx
import argparse
from src.Module import ImageEncoder
from src.Module import MemAttention
from src.Module import MemEncoder
from src.Module import MaskDecoder
from sam2.build_sam import build_sam2

def export_image_encoder(model,onnx_path, batch_size=3):
    print(">>> Exporting Image Encoder...")
    input_img = torch.randn(batch_size, 3, 256, 256).cpu()
    out = model(input_img)
    output_names = ["pix_feat","high_res_feat0","high_res_feat1","vision_feats","vision_pos_embed"]
    torch.onnx.export(
        model,
        input_img,
        onnx_path+"image_encoder.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=output_names,
    )
    onnx_model = onnx.load(onnx_path+"image_encoder.onnx")
    onnx.checker.check_model(onnx_model)
    print(f"[SUCCESS] Image Encoder (Batch {batch_size}) exported successfully!")


def export_memory_attention(model,onnx_path, batch_size=3):
    print(">>> Exporting Memory Attention...")
    current_vision_feat = torch.randn(batch_size,256,16,16)      #[batch_size, 256, 16, 16]
    current_vision_pos_embed = torch.randn(256,batch_size,256)  #[256, batch_size, 256]
    memory_0 = torch.randn(batch_size,16,256) # [batch size, num obj ptr, feature size]
    memory_1 = torch.randn(batch_size,7,64,16,16)
    memory_pos_embed = torch.randn(batch_size,7*256+64,64)      #[y*256,1,64]
    cond_frame_id_diff = torch.tensor(10.0)
    out = model(
            current_vision_feat = current_vision_feat,
            current_vision_pos_embed = current_vision_pos_embed,
            memory_0 = memory_0,
            memory_1 = memory_1,
            memory_pos_embed = memory_pos_embed,
            cond_frame_id_diff = cond_frame_id_diff,
        )
    input_name = ["current_vision_feat",
                "current_vision_pos_embed",
                "memory_0",
                "memory_1",
                "memory_pos_embed",
                "cond_frame_id_diff",]
    # For MemAttention, keep number of memories dynamic, but batch size static
    dynamic_axes = {
        "memory_0": {1: "num"},
        "memory_1": {1: "buff_size"},
        "memory_pos_embed": {1: "buff_size_embed"}
    }
    torch.onnx.export(
        model,
        (current_vision_feat,current_vision_pos_embed,memory_0,memory_1,memory_pos_embed,cond_frame_id_diff),
        onnx_path+"memory_attention.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=["image_embed"],
        dynamic_axes = dynamic_axes
    )
    onnx_model = onnx.load(onnx_path+"memory_attention.onnx")
    onnx.checker.check_model(onnx_model)
    print(f"[SUCCESS] Memory Attention (Batch {batch_size}) exported successfully!")


def export_mask_decoder(model,onnx_path, batch_size=3):
    print(">>> Exporting Mask Decoder...")
    point_coords = torch.randn(batch_size,2,2).cpu()
    point_labels = torch.randn(batch_size,2).cpu()
    image_embed = torch.randn(batch_size,256,16,16).cpu()
    high_res_feats_0 = torch.randn(batch_size,32,64,64).cpu()
    high_res_feats_1 = torch.randn(batch_size,64,32,32).cpu()

    out = model(
        point_coords = point_coords,
        point_labels = point_labels,
        image_embed = image_embed,
        high_res_feats_0 = high_res_feats_0,
        high_res_feats_1 = high_res_feats_1
    )
    input_name = ["point_coords","point_labels","image_embed","high_res_feats_0","high_res_feats_1"]
    output_name = ["obj_ptr","mask_for_mem","pred_mask", "iou", "occ_logit"]
    # Point coords can be dynamic in points count, but batch is static
    dynamic_axes = {
        "point_coords":{1:"num_points"},
        "point_labels": {1:"num_points"},
    }
    torch.onnx.export(
        model,
        (point_coords,point_labels,image_embed,high_res_feats_0,high_res_feats_1),
        onnx_path+"mask_decoder.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=output_name,
        dynamic_axes = dynamic_axes
    )
    onnx_model = onnx.load(onnx_path+"mask_decoder.onnx")
    onnx.checker.check_model(onnx_model)
    print(f"[SUCCESS] Mask Decoder (Batch {batch_size}) exported successfully!")


def export_memory_encoder(model,onnx_path, batch_size=3):
    print(">>> Exporting Memory Encoder...")
    mask_for_mem = torch.randn(batch_size,1,256,256)
    pix_feat = torch.randn(batch_size,256,16,16)
    occ_logit = torch.randn(batch_size,1)

    out = model(mask_for_mem = mask_for_mem,pix_feat = pix_feat,occ_logit = occ_logit)

    input_names = ["mask_for_mem","pix_feat","occ_logit"]
    output_names = ["maskmem_features","maskmem_pos_enc","temporal_code"]
    torch.onnx.export(
        model,
        (mask_for_mem,pix_feat,occ_logit),
        onnx_path+"memory_encoder.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_names,
        output_names= output_names,
    )
    onnx_model = onnx.load(onnx_path+"memory_encoder.onnx")
    onnx.checker.check_model(onnx_model)
    print(f"[SUCCESS] Memory Encoder (Batch {batch_size}) exported successfully!")



if __name__ == "__main__":
    model_type = "tiny"
    outdir = "/home/jianwei/vsim/unitree_g1_hw_endpoint/image_processing/sam2_onnx/tiny_batch3_256/"
    config = "configs/sam2.1/sam2.1_hiera_t_256.yaml"
    checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
    batch_size = 3

    os.makedirs(outdir, exist_ok=True)

    sam2_model = build_sam2(config, checkpoint, device="cpu")

    image_encoder = ImageEncoder(sam2_model).cpu()
    export_image_encoder(image_encoder,outdir, batch_size)

    mask_decoder = MaskDecoder(sam2_model).cpu()
    export_mask_decoder(mask_decoder,outdir, batch_size)

    mem_attention = MemAttention(sam2_model).cpu()
    export_memory_attention(mem_attention,outdir, batch_size)

    mem_encoder   = MemEncoder(sam2_model).cpu()
    export_memory_encoder(mem_encoder,outdir, batch_size)
