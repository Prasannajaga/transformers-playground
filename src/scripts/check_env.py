import torch
import os
import sys

def check_flash_attention():
    """Checks if Flash Attention is available via torch or flash_attn package."""
    print("\n[CHECK] Flash Attention Availability")
    
    # Check PyTorch's built-in SDPA (Scaled Dot Product Attention)
    try:
        # Create dummy tensors on CUDA if available
        if torch.cuda.is_available():
            q = torch.randn(1, 12, 8, 64, device='cuda', dtype=torch.float16)
            k = torch.randn(1, 12, 8, 64, device='cuda', dtype=torch.float16)
            v = torch.randn(1, 12, 8, 64, device='cuda', dtype=torch.float16)
            
            # Check if SDPA supports flash attention backend
            # Note: This is more of a runtime check, but we can inspect available backends
            from torch.backends import cuda
            if hasattr(cuda, 'flash_sdp_enabled'):
                print(f"  - torch.backends.cuda.flash_sdp_enabled(): {cuda.flash_sdp_enabled()}")
            
            # Check via flash_attn package if installed
            try:
                import flash_attn
                print(f"  - flash_attn package version: {flash_attn.__version__}")
                print("  - flash_attn is importable: YES")
            except ImportError:
                print("  - flash_attn package check: Not installed (using PyTorch SDPA?)")
                
        else:
            print("  - CUDA not available, cannot check Flash Attention hardware support.")
            
    except Exception as e:
        print(f"  - Error checking Flash Attention: {e}")

def check_distributed_gpus():
    """Checks for distributed GPU support and availability."""
    print("\n[CHECK] Distributed GPU Support")
    
    if not torch.cuda.is_available():
        print("  - CUDA is NOT available.")
        return

    device_count = torch.cuda.device_count()
    print(f"  - CUDA Device Count: {device_count}")
    
    for i in range(device_count):
        print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")

    # Check distributed support
    if torch.distributed.is_available():
        print("  - torch.distributed is available: YES")
        if torch.distributed.is_nccl_available():
            print("  - NCCL backend available: YES")
        else:
            print("  - NCCL backend available: NO")
            
        if torch.distributed.is_gloo_available():
            print("  - Gloo backend available: YES")
        else:
             print("  - Gloo backend available: NO")
    else:
        print("  - torch.distributed is available: NO")

    # Check environment variables often used in Vertex AI/Distributed jobs
    print("\n[CHECK] Distributed Environment Variables")
    rank = os.environ.get('RANK')
    world_size = os.environ.get('WORLD_SIZE')
    local_rank = os.environ.get('LOCAL_RANK')
    master_addr = os.environ.get('MASTER_ADDR')
    master_port = os.environ.get('MASTER_PORT')
    
    print(f"  - RANK: {rank}")
    print(f"  - WORLD_SIZE: {world_size}")
    print(f"  - LOCAL_RANK: {local_rank}")
    print(f"  - MASTER_ADDR: {master_addr}")
    print(f"  - MASTER_PORT: {master_port}")

def main():
    print("="*60)
    print("VERTEX AI / ENVIRONMENT INSPECTION SCRIPT")
    print("="*60)
    
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    check_flash_attention()
    check_distributed_gpus()
    
    print("\n" + "="*60)
    print("Inspection Complete")
    print("="*60)

if __name__ == "__main__":
    main()
