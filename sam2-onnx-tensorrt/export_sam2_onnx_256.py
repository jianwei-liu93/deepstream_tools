#!/usr/bin/env python3

import os
import torch
import onnx
import argparse
from src.Module import ImageEncoder
from src.Module import MemAttention
from src.Module import MemEncoder
from src.Module import MaskDecoder
from sam2.build_sam import build_sam2
from onnx.shape_inference import infer_shapes

def as_tensorrt_compatible(onnx_path):
    print(f">>> Making {onnx_path} TensorRT compatible...")
    model = onnx.load(onnx_path)
    # Run shape inference
    model = infer_shapes(model)
    onnx.save(model, onnx_path)
    print(f"[SUCCESS] {onnx_path} is now TensorRT compatible.")



def export_image_encoder(model,onnx_path):
    batch_size = 1
    
    print("\n>>> Exporting Image Encoder...")
    input_img = torch.randn(batch_size, 3, 256, 256, dtype=torch.float32).cpu()
    out = model(input_img)
    output_names = ["pix_feat","high_res_feat0","high_res_feat1","vision_feats","vision_pos_embed"]
    dynamic_axes = {}
    torch.onnx.export(
        model,
        input_img,
        onnx_path+"image_encoder.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    onnx_model = onnx.load(onnx_path+"image_encoder.onnx")
    onnx.checker.check_model(onnx_model)
    as_tensorrt_compatible(onnx_path+"image_encoder.onnx")
    print("[SUCCESS] Image Encoder exported successfully!")


def export_memory_attention(model,onnx_path):
    print(">>> Exporting Memory Attention...")
    batch_size = 1
    num_obj_ptr = 16
    num_mask = 7 
    
    current_vision_feat = torch.randn(batch_size, 256, 16, 16, dtype=torch.float32).cpu()
    current_vision_pos_embed = torch.randn(256, batch_size, 256, dtype=torch.float32).cpu()
    memory_0 = torch.randn(batch_size, num_obj_ptr, 256, dtype=torch.float32).cpu()
    memory_1 = torch.randn(batch_size, num_mask, 64, 16, 16, dtype=torch.float32).cpu()
    memory_pos_embed = torch.randn(batch_size, num_mask*256+4*num_obj_ptr, 64, dtype=torch.float32).cpu()      #[1,y*256,64]
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
    dynamic_axes = {
        "memory_0": {1: "num"},
        "memory_1": {1: "buff_size"},
        "memory_pos_embed": {1: "buff_size_embed"},
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
    as_tensorrt_compatible(onnx_path+"memory_attention.onnx")
    print("[SUCCESS] Memory Attention exported successfully!")


def export_mask_decoder(model,onnx_path):
    print(">>> Exporting Mask Decoder...")
    batch_size = 1
    point_coords = torch.randn(batch_size,2,2).cpu()
    point_labels = torch.randn(batch_size,2).cpu()
    test_image_embed = torch.randn(batch_size, 256, 16, 16, dtype=torch.float32).cpu()
    test_high_res_feats_0 = torch.randn(batch_size, 32, 64, 64, dtype=torch.float32).cpu()
    test_high_res_feats_1 = torch.randn(batch_size, 64, 32, 32, dtype=torch.float32).cpu()

    out = model(
        point_coords = point_coords,
        point_labels = point_labels,
        image_embed = test_image_embed,
        high_res_feats_0 = test_high_res_feats_0,
        high_res_feats_1 = test_high_res_feats_1
    )
    input_name = ["point_coords","point_labels","image_embed","high_res_feats_0","high_res_feats_1"]
    output_name = ["obj_ptr","mask_for_mem","pred_mask", "iou", "occ_logit"]
    dynamic_axes = {
        "point_coords": {1:"num_points"},
        "point_labels": {1:"num_points"},
    }
    torch.onnx.export(
        model,
        (point_coords,point_labels,test_image_embed,test_high_res_feats_0,test_high_res_feats_1),
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
    as_tensorrt_compatible(onnx_path+"mask_decoder.onnx")
    print("[SUCCESS] Mask Decoder exported successfully!")


def export_memory_encoder(model,onnx_path):
    print(">>> Exporting Memory Encoder...")
    batch_size = 1
    test_mask_for_mem = torch.randn(batch_size, 1, 256, 256, dtype=torch.float32).cpu()
    test_pix_feat = torch.randn(batch_size, 256, 16, 16, dtype=torch.float32).cpu()
    occ_logit = torch.randn(batch_size, 1)
    dynamic_axes = {}

    out = model(mask_for_mem = test_mask_for_mem,pix_feat = test_pix_feat,occ_logit = occ_logit)

    input_names = ["mask_for_mem","pix_feat","occ_logit"]
    output_names = ["maskmem_features","maskmem_pos_enc","temporal_code"]
    dynamic_axes = {}
    torch.onnx.export(
        model,
        (test_mask_for_mem,test_pix_feat,occ_logit),
        onnx_path+"memory_encoder.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_names,
        output_names= output_names,
        dynamic_axes = dynamic_axes
    )
    onnx_model = onnx.load(onnx_path+"memory_encoder.onnx")
    onnx.checker.check_model(onnx_model)
    as_tensorrt_compatible(onnx_path+"memory_encoder.onnx")
    print("[SUCCESS] Memory Encoder exported successfully!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="export SAM2.1 to onnx")
    parser.add_argument("--model",type=str,choices=["tiny", "small", "base_plus", "large"],
        default="tiny",required=False,help="SAM2 model type. Choose one of: tiny, small, base_plus, large")
    args = parser.parse_args()

    model_type = args.model
    # Output natively generated models to the single batch tiny_256 directory
    outdir = "/home/jianwei/vsim/unitree_g1_hw_endpoint/image_processing/sam2_onnx/tiny_256/"
    config = "configs/sam2.1/sam2.1_hiera_t_256.yaml"
    checkpoint = "checkpoints/sam2.1_hiera_{}.pt".format(model_type)

    os.makedirs(outdir, exist_ok=True)

    sam2_model = build_sam2(config, checkpoint, device="cpu")

    image_encoder = ImageEncoder(sam2_model).cpu()
    export_image_encoder(image_encoder,outdir)

    mask_decoder = MaskDecoder(sam2_model).cpu()
    export_mask_decoder(mask_decoder,outdir)

    mem_attention = MemAttention(sam2_model).cpu()
    export_memory_attention(mem_attention,outdir)

    mem_encoder   = MemEncoder(sam2_model).cpu()
    export_memory_encoder(mem_encoder,outdir)
