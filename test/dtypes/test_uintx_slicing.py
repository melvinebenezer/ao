import torch
from torchao.dtypes.uintx.uintx import UintxTensor, to_uintx

def test_uintx_slicing():
    original_data = torch.tensor([10, 25, 40, 55, 5, 20, 35, 50], dtype=torch.uint8)
    
    # Convert to UintxTensor (6-bit)
    uintx_tensor = to_uintx(original_data, torch.uint6)
    
    # Perform slicing
    sliced_uintx = uintx_tensor[2:6]
    
    # Convert back to regular tensor for comparison
    sliced_data = sliced_uintx.get_plain()
    
    # Expected result
    expected_slice = torch.tensor([40, 55, 5, 20], dtype=torch.uint8)
    
    # Check if the slicing worked correctly
    if torch.all(sliced_data == expected_slice):
        print("Slicing test passed!")
    else:
        print("Slicing test failed.")
        print(f"Expected: {expected_slice}")
        print(f"Got: {sliced_data}")

    # Additional test: slice with step
    step_sliced_uintx = uintx_tensor[1::2]
    step_sliced_data = step_sliced_uintx.get_plain()
    expected_step_slice = torch.tensor([25, 55, 20, 50], dtype=torch.uint8)
    
    if torch.all(step_sliced_data == expected_step_slice):
        print("Step slicing test passed!")
    else:
        print("Step slicing test failed.")
        print(f"Expected: {expected_step_slice}")
        print(f"Got: {step_sliced_data}")

if __name__ == "__main__":
    test_uintx_slicing()