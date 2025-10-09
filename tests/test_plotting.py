import numpy as np
import matplotlib as mpl
import pytest
from unittest.mock import Mock, MagicMock

# Set a non-interactive backend for matplotlib before importing pyplot or other plotting modules
mpl.use('Agg')
import matplotlib.pyplot as plt # Import after setting backend

from common_utils.plotting import (
    colorblind_cycle,
    setup_matplotlib,
    set_size,
    generate_diverse_colors,
    legend_without_duplicate_labels
)

# Test colorblind_cycle
def test_colorblind_cycle_defined():
    assert isinstance(colorblind_cycle, list)
    assert len(colorblind_cycle) > 0
    for color_hex in colorblind_cycle:
        assert isinstance(color_hex, str)
        assert color_hex.startswith('#')
        assert len(color_hex) == 7 # e.g., #RRGGBB

# Pytest fixture to save and restore rcParams
@pytest.fixture(scope="function")
def mpl_rc_params_restorer():
    original_params = mpl.rcParams.copy()
    yield
    mpl.rcParams.update(original_params)

# Test setup_matplotlib
def test_setup_matplotlib(mpl_rc_params_restorer): # Use the fixture
    # Call the setup function - it takes no arguments as per common_utils/plotting.py
    setup_matplotlib()

    # Check if specific rcParams were set as expected by the function
    assert mpl.rcParams["text.usetex"] is True
    assert mpl.rcParams["font.family"] == ["serif"] # rcParams stores font.family as a list
    assert mpl.rcParams["axes.labelsize"] == 10
    assert mpl.rcParams["font.size"] == 10
    assert mpl.rcParams["legend.fontsize"] == 8
    assert mpl.rcParams["xtick.labelsize"] == 7
    assert mpl.rcParams["ytick.labelsize"] == 7

# Test set_size
def test_set_size():
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2

    # Test case 1: thesis
    width_pt_thesis = 426.79135
    fraction = 1
    subplots = (1, 1)
    expected_width_thesis = (width_pt_thesis * fraction) * inches_per_pt
    expected_height_thesis = expected_width_thesis * golden_ratio * (subplots[0] / subplots[1])
    assert set_size("thesis", fraction=fraction, subplots=subplots) == pytest.approx((expected_width_thesis, expected_height_thesis))

    # Test case 2: beamer, fraction, subplots
    width_pt_beamer = 307.28987
    fraction = 0.5
    subplots = (2, 1)
    expected_width_beamer = (width_pt_beamer * fraction) * inches_per_pt
    expected_height_beamer = expected_width_beamer * golden_ratio * (subplots[0] / subplots[1])
    assert set_size("beamer", fraction=fraction, subplots=subplots) == pytest.approx((expected_width_beamer, expected_height_beamer))

    # Test case 3: custom width
    custom_width_pt = 500
    fraction = 1
    subplots = (1, 2)
    expected_width_custom = (custom_width_pt * fraction) * inches_per_pt
    expected_height_custom = expected_width_custom * golden_ratio * (subplots[0] / subplots[1])
    assert set_size(custom_width_pt, fraction=fraction, subplots=subplots) == pytest.approx((expected_width_custom, expected_height_custom))

    # Test with invalid doc_type string that cannot be converted to float
    with pytest.raises(TypeError): # The function will try width_pt * fraction, causing TypeError
        set_size("invalid_type_string")

# Test generate_diverse_colors
def test_generate_diverse_colors_number_output():
    colors = generate_diverse_colors(5)
    assert isinstance(colors, np.ndarray)
    assert colors.shape == (5, 3)
    assert np.all(colors >= 0)
    assert np.all(colors <= 1)

def test_generate_diverse_colors_with_ref():
    ref_color_hex = "#FF0000" # Red
    # The function expects ref_color to be hex if it's one of its predefined colors
    # or it can be an RGB tuple/array. Let's test with a hex string it knows.
    # One of the predefined colors is 'e6194b' (a shade of red)
    # If we provide a ref_color that's in its list, it should try to pop it.
    # For this test, let's use a color that is in its `colors_hex` list.
    # The internal `colors_hex` has 64 colors.

    # Pick a color from the internal list to use as ref_color
    # (Assuming common_utils.plotting.colors_hex is accessible or we use a known one)
    # For robustness, let's not assume access to internal `colors_hex`.
    # The function converts hex to RGB. Let's test with an RGB ref_color.
    ref_rgb = np.array([1.0, 0.0, 0.0]) # Pure Red
    num_to_generate = 5
    colors = generate_diverse_colors(num_to_generate, ref_color=ref_rgb)
    assert colors.shape == (num_to_generate, 3)

    # Check that the generated colors are not identical to the reference color
    # This is a basic check; true diversity is harder to quantify here.
    for color in colors:
        assert not np.allclose(color, ref_rgb, atol=1e-2) # Check not too close

    # Test when n_colors is 1 and ref_color is provided
    colors_one = generate_diverse_colors(1, ref_color=ref_rgb)
    assert colors_one.shape == (1,3)
    assert not np.allclose(colors_one[0], ref_rgb, atol=1e-2)


def test_generate_diverse_colors_more_than_predefined():
    # The internal list `colors_hex` has 64 colors.
    # Let's test generating more than that to ensure it falls back to HUSL.
    num_colors = 70
    colors = generate_diverse_colors(num_colors)
    assert colors.shape == (num_colors, 3)
    assert np.all(colors >= 0) and np.all(colors <= 1)
    # Ensure unique colors, at least for a reasonable number
    unique_colors = np.unique(colors, axis=0)
    assert unique_colors.shape[0] >= min(num_colors, 64) # At least the predefined ones should be unique. HUSL might produce very close ones.

def test_generate_diverse_colors_zero():
    colors = generate_diverse_colors(0)
    assert isinstance(colors, np.ndarray)
    assert colors.shape == (0,) # Returns np.array([]) which has shape (0,)


# Test legend_without_duplicate_labels
def test_legend_without_duplicate_labels_basic():
    # Using MagicMock for Line2D objects as they might have a 'get_label' method internally called by some plt versions
    h1, h2, h3 = MagicMock(spec=plt.Line2D), MagicMock(spec=plt.Line2D), MagicMock(spec=plt.Line2D)
    h1.get_label.return_value = "A" # Mocking this for consistency if `ax.legend()` was called
    h2.get_label.return_value = "B"
    h3.get_label.return_value = "C"

    handles = [h1, h2, h1, h3, h2]
    labels = ["A", "B", "A", "C", "B"]

    mock_ax = Mock()
    mock_ax.get_legend_handles_labels.return_value = (handles, labels)

    unique_handles, unique_labels = legend_without_duplicate_labels(mock_ax)

    assert list(unique_labels) == ["A", "B", "C"] # Convert tuple to list for comparison
    assert list(unique_handles) == [h1, h2, h3]

def test_legend_without_duplicate_labels_no_duplicates():
    h1, h2, h3 = MagicMock(spec=plt.Line2D), MagicMock(spec=plt.Line2D), MagicMock(spec=plt.Line2D)
    h1.get_label.return_value = "A"
    h2.get_label.return_value = "B"
    h3.get_label.return_value = "C"

    handles = [h1, h2, h3]
    labels = ["A", "B", "C"]

    mock_ax = Mock()
    mock_ax.get_legend_handles_labels.return_value = (handles, labels)

    unique_handles, unique_labels = legend_without_duplicate_labels(mock_ax)

    assert list(unique_labels) == ["A", "B", "C"]
    assert list(unique_handles) == [h1, h2, h3]

def test_legend_without_duplicate_labels_empty():
    mock_ax = Mock()
    mock_ax.get_legend_handles_labels.return_value = ([], [])

    # The current implementation raises ValueError for empty inputs
    with pytest.raises(ValueError, match="not enough values to unpack"):
        legend_without_duplicate_labels(mock_ax)

def test_legend_without_duplicate_labels_custom_ax_call():
    # Test if a custom ax is passed to ax.legend() if needed
    h1 = MagicMock(spec=plt.Line2D); h1.get_label.return_value = "A"
    handles = [h1, h1]; labels = ["A", "A"]

    fig, ax = plt.subplots()
    ax.plot([0,1],[0,1], label="A") # This will create a handle and label
    ax.plot([0,1],[1,0], label="A") # This will create another

    # The function internally calls ax.get_legend_handles_labels()
    # No, the function takes an ax where get_legend_handles_labels() can be called.
    # It does not call ax.legend() itself.

    unique_handles, unique_labels = legend_without_duplicate_labels(ax)

    assert len(unique_labels) == 1
    assert unique_labels[0] == "A"
    assert len(unique_handles) == 1
    plt.close(fig) # Close the figure to free memory
