import torch
import pytest
from torchao.dtypes.uintx.uintx import UintxTensor, UintxLayoutType, UintxAQTLayout, to_uintx

def test_uintx_basic():
    # Test UintxTensor
    original_data = torch.tensor([10, 25, 40, 55, 5, 20, 35, 50], dtype=torch.uint8)
    uintx_tensor = to_uintx(original_data, torch.uint6)  # 6-bit UintxTensor
    
    # Verify packing and unpacking
    unpacked_data = uintx_tensor.get_plain()
    assert torch.all(original_data == unpacked_data), "Packing/unpacking failed"

    # Test UintxLayoutType
    layout_type = UintxLayoutType(dtype=torch.uint6, pack_dim=-1)
    processed_tensor = layout_type.post_process(original_data)
    
    assert isinstance(processed_tensor, UintxTensor), "Layout application failed"
    assert torch.all(processed_tensor.get_plain() == original_data), "Layout processing changed data"

    # Test UintxAQTLayout
    scale = torch.tensor([0.5])
    zero_point = torch.tensor([0], dtype=torch.int32)
    
    aqt_layout = UintxAQTLayout.from_plain(original_data, scale, zero_point, layout_type)
    
    # Verify quantization
    int_data, layout_scale, layout_zero_point = aqt_layout.get_plain()
    assert torch.all(int_data == original_data), "Int data should be unchanged"
    assert torch.all(layout_scale == scale), "Scale should be unchanged"
    assert torch.all(layout_zero_point == zero_point), "Zero point should be unchanged"

    # Test that the layout_type is correctly stored
    assert isinstance(aqt_layout.layout_type, UintxLayoutType), "Layout type should be UintxLayoutType"
    assert aqt_layout.layout_type.dtype == torch.uint6, "Layout type should have correct dtype"
    assert aqt_layout.layout_type.pack_dim == -1, "Layout type should have correct pack_dim"

    print("All tests passed successfully!")

if __name__ == "__main__":
    test_uintx_basic()