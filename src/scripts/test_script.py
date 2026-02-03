#!/usr/bin/env python3
"""
GPU and PyTorch Information Test Script
Prints comprehensive information about PyTorch installation and available devices
"""

import torch
import sys


def print_section(title: str):
    """Print a formatted section heade  r"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_pytorch_info():
    """Print PyTorch version and build information"""
    print_section("PyTorch Information")
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Build: {torch.version.git_version}")
    
    # Check if compiled with CUDA
    print(f"\nBuilt with CUDA: {torch.version.cuda is not None}")
    if torch.version.cuda:
        print(f"CUDA Version (Build): {torch.version.cuda}")
    
    # Check if compiled with cuDNN
    if hasattr(torch.backends, 'cudnn'):
        print(f"cuDNN Available: {torch.backends.cudnn.is_available()}")
        if torch.backends.cudnn.is_available():
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")


def print_cuda_info():
    """Print CUDA availability and device information"""
    print_section("CUDA Information")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version (Runtime): {torch.version.cuda}")
        print(f"Number of CUDA Devices: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"Default CUDA Device Name: {torch.cuda.get_device_name()}")
        
        # Print information for each CUDA device
        for i in range(torch.cuda.device_count()):
            print(f"\n--- Device {i} ---")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            
            # Get device properties
            props = torch.cuda.get_device_properties(i)
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Multi-Processor Count: {props.multi_processor_count}")
            
            # Memory information
            print(f"  Allocated Memory: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Reserved Memory: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(f"  Free Memory: {(props.total_memory - torch.cuda.memory_allocated(i)) / 1024**3:.2f} GB")
    else:
        print("No CUDA devices available")
        print("\nPossible reasons:")
        print("  - No NVIDIA GPU installed")
        print("  - CUDA drivers not installed")
        print("  - PyTorch installed without CUDA support")


def print_device_info():
    """Print information about all available devices"""
    print_section("Available Devices")
    
    # CPU
    print("CPU:")
    print(f"  Device: cpu")
    print(f"  Available: True")
    
    # CUDA
    if torch.cuda.is_available():
        print("\nCUDA Devices:")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: cuda:{i} ({torch.cuda.get_device_name(i)})")
    
    # MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("\nMPS (Apple Silicon):")
        print(f"  Device: mps")
        print(f"  Available: True")
    
    # Recommended device
    print("\nRecommended Device:")
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"  {device}")


def test_tensor_operations():
    """Test basic tensor operations on available devices"""
    print_section("Tensor Operations Test")
    
    # Test on CPU
    print("Testing on CPU...")
    try:
        cpu_tensor = torch.randn(1000, 1000)
        cpu_result = torch.matmul(cpu_tensor, cpu_tensor)
        print(f"  ✓ CPU tensor operation successful")
        print(f"    Tensor shape: {cpu_result.shape}")
        print(f"    Tensor device: {cpu_result.device}")
    except Exception as e:
        print(f"  ✗ CPU tensor operation failed: {e}")
    
    # Test on CUDA if available
    if torch.cuda.is_available():
        print("\nTesting on CUDA...")
        try:
            cuda_tensor = torch.randn(1000, 1000, device='cuda')
            cuda_result = torch.matmul(cuda_tensor, cuda_tensor)
            torch.cuda.synchronize()
            print(f"  ✓ CUDA tensor operation successful")
            print(f"    Tensor shape: {cuda_result.shape}")
            print(f"    Tensor device: {cuda_result.device}")
        except Exception as e:
            print(f"  ✗ CUDA tensor operation failed: {e}")


def print_additional_info():
    """Print additional useful information"""
    print_section("Additional Information")
    
    # OpenMP
    print(f"OpenMP Available: {torch.backends.openmp.is_available()}")
    
    # MKL
    if hasattr(torch.backends, 'mkl'):
        print(f"MKL Available: {torch.backends.mkl.is_available()}")
    
    # Number of threads
    print(f"Number of Threads: {torch.get_num_threads()}")
    print(f"Number of Inter-op Threads: {torch.get_num_interop_threads()}")
    
    # Deterministic mode
    print(f"Deterministic Mode: {torch.are_deterministic_algorithms_enabled()}")
    
    # Autograd
    print(f"Autograd Enabled: {torch.is_grad_enabled()}")


def main():
    """Main function to run all tests"""
    print("\n" + "=" * 60)
    print("  PyTorch GPU Test Script")
    print("=" * 60)
    
    print_pytorch_info()
    print_cuda_info()
    print_device_info()
    test_tensor_operations()
    print_additional_info()
    
    print("\n" + "=" * 60)
    print("  Test Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
