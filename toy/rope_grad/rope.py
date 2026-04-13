"""
RoPE (Rotary Position Embeddings) - Educational Implementation
A simple, visual exploration of how RoPE works
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
import seaborn as sns

# Set up nice plotting defaults
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# =============================================================================
# CORE ROPE IMPLEMENTATION (2D)
# =============================================================================

def rotate_2d(x, y, angle):
    """
    Rotate a 2D point (x, y) by a given angle.
    This is the core operation in RoPE!
    
    Args:
        x, y: 2D coordinates
        angle: rotation angle in radians
    
    Returns:
        rotated (x', y') coordinates
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a
    
    return x_rot, y_rot


def apply_rope_2d(x, y, position, theta=10000.0):
    """
    Apply RoPE to a single 2D point at a given position.
    
    The magic formula: angle = position / theta
    
    Args:
        x, y: embedding coordinates (one dimension pair)
        position: token position in sequence
        theta: base frequency (higher = slower rotation)
    
    Returns:
        rotated coordinates
    """
    angle = position / theta
    return rotate_2d(x, y, angle)


def apply_rope_nd(embeddings, positions, theta_base=10000.0):
    """
    Apply RoPE to full embeddings (multiple dimension pairs).
    
    Args:
        embeddings: shape (seq_len, d_model) where d_model is even
        positions: shape (seq_len,) - position indices
        theta_base: base for frequency calculation
    
    Returns:
        embeddings with RoPE applied
    """
    seq_len, d_model = embeddings.shape
    assert d_model % 2 == 0, "Embedding dimension must be even"
    
    result = embeddings.copy()
    
    # Each pair of dimensions gets a different frequency
    for i in range(d_model // 2):
        # Frequency decreases for higher dimensions
        theta = theta_base ** (2 * i / d_model)
        
        for pos_idx, pos in enumerate(positions):
            angle = pos / theta
            
            x = embeddings[pos_idx, 2*i]
            y = embeddings[pos_idx, 2*i + 1]
            
            x_rot, y_rot = rotate_2d(x, y, angle)
            
            result[pos_idx, 2*i] = x_rot
            result[pos_idx, 2*i + 1] = y_rot
    
    return result


# =============================================================================
# VISUALIZATION 1: CLOCK ANALOGY
# =============================================================================

def visualize_clock_analogy(positions=[0, 5, 10, 20], theta_values=[10, 50, 100], 
                           figsize=(15, 5)):
    """
    Visualize RoPE as clocks rotating at different speeds.
    
    Each dimension pair = one clock
    Different theta = different rotation speeds
    """
    fig, axes = plt.subplots(1, len(theta_values), figsize=figsize)
    if len(theta_values) == 1:
        axes = [axes]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))
    
    for ax_idx, theta in enumerate(theta_values):
        ax = axes[ax_idx]
        
        # Draw clock circle
        circle = Circle((0, 0), 1, fill=False, color='gray', linestyle='--', linewidth=1)
        ax.add_patch(circle)
        
        # Draw clock hands for each position
        for pos_idx, pos in enumerate(positions):
            angle = pos / theta
            
            # Clock hand (unit vector rotated by angle)
            x = np.cos(angle)
            y = np.sin(angle)
            
            arrow = FancyArrowPatch((0, 0), (x, y),
                                   arrowstyle='->', 
                                   mutation_scale=20,
                                   linewidth=2,
                                   color=colors[pos_idx],
                                   label=f'pos={pos}')
            ax.add_patch(arrow)
        
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.set_title(f'θ = {theta}\n({"fast" if theta < 50 else "slow"} rotation)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        
        if ax_idx == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=8)
    
    fig.suptitle('🕐 Clock Analogy: Each dimension pair rotates like a clock hand', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# VISUALIZATION 2: 2D SCATTER WITH SPIRAL
# =============================================================================

def visualize_2d_spiral(max_position=50, theta=10000.0, figsize=(12, 5)):
    """
    Plot how a single dimension pair evolves as position increases.
    Should see a spiral/circle pattern!
    """
    positions = np.arange(max_position)
    
    # Start with a random 2D point
    x0, y0 = 1.0, 0.5
    
    # Apply RoPE to create trajectory
    points = []
    for pos in positions:
        x_rot, y_rot = apply_rope_2d(x0, y0, pos, theta)
        points.append([x_rot, y_rot])
    
    points = np.array(points)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Trajectory with color gradient
    colors = plt.cm.plasma(np.linspace(0, 1, len(positions)))
    
    for i in range(len(positions) - 1):
        ax1.plot(points[i:i+2, 0], points[i:i+2, 1], 
                color=colors[i], linewidth=2, alpha=0.7)
    
    # Mark start and end
    ax1.scatter(points[0, 0], points[0, 1], s=200, c='green', 
               marker='o', edgecolor='black', linewidth=2, 
               label='Start (pos=0)', zorder=5)
    ax1.scatter(points[-1, 0], points[-1, 1], s=200, c='red', 
               marker='s', edgecolor='black', linewidth=2,
               label=f'End (pos={max_position-1})', zorder=5)
    
    ax1.set_xlabel('x dimension', fontsize=12)
    ax1.set_ylabel('y dimension', fontsize=12)
    ax1.set_title('2D Embedding Trajectory (colored by position)', 
                 fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Right plot: Angle vs position
    angles = positions / theta
    ax2.plot(positions, angles, linewidth=2, color='purple')
    ax2.set_xlabel('Position', fontsize=12)
    ax2.set_ylabel('Rotation Angle (radians)', fontsize=12)
    ax2.set_title(f'Rotation Angle vs Position (θ={theta})', 
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add horizontal lines for 2π multiples
    for n in range(1, int(angles[-1] / (2*np.pi)) + 2):
        ax2.axhline(y=n*2*np.pi, color='red', linestyle='--', 
                   alpha=0.5, linewidth=1)
        ax2.text(max_position * 0.02, n*2*np.pi, f'{n}×2π', 
                fontsize=9, color='red')
    
    plt.tight_layout()
    return fig


# =============================================================================
# VISUALIZATION 3: ATTENTION MATRIX COMPARISON
# =============================================================================

def compute_attention_scores(embeddings):
    """
    Compute attention scores (simplified - just dot products normalized).
    
    Args:
        embeddings: shape (seq_len, d_model)
    
    Returns:
        attention matrix: shape (seq_len, seq_len)
    """
    # Compute dot products (Q @ K^T)
    attention = embeddings @ embeddings.T
    
    # Normalize to [0, 1] for visualization
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    
    return attention


def visualize_attention_comparison(seq_len=20, d_model=64, theta=10000.0, 
                                  figsize=(15, 5)):
    """
    Compare attention patterns with and without RoPE.
    
    This shows the key insight: RoPE makes nearby tokens attend to each other more!
    """
    positions = np.arange(seq_len)
    
    # Random embeddings (no positional info)
    embeddings_no_pos = np.random.randn(seq_len, d_model) * 0.1
    
    # Apply RoPE
    embeddings_with_rope = apply_rope_nd(embeddings_no_pos.copy(), positions, theta)
    
    # Compute attention matrices
    attn_no_pos = compute_attention_scores(embeddings_no_pos)
    attn_with_rope = compute_attention_scores(embeddings_with_rope)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # No positional encoding
    im1 = axes[0].imshow(attn_no_pos, cmap='YlOrRd', aspect='auto')
    axes[0].set_title('❌ No Positional Encoding\n(random patterns)', 
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # With RoPE
    im2 = axes[1].imshow(attn_with_rope, cmap='YlOrRd', aspect='auto')
    axes[1].set_title('✅ With RoPE\n(diagonal pattern!)', 
                     fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Difference
    diff = attn_with_rope - attn_no_pos
    im3 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto', 
                        vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    axes[2].set_title('Difference\n(blue=RoPE increases attention)', 
                     fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Key Position')
    axes[2].set_ylabel('Query Position')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    fig.suptitle('Attention Patterns: RoPE creates locality bias', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# VISUALIZATION 4: MANUAL WALKTHROUGH
# =============================================================================

def manual_walkthrough(positions=[0, 5, 10], theta=100.0):
    """
    Step-by-step manual calculation showing exactly what happens.
    Pick 3 tokens and show all the math!
    """
    print("=" * 70)
    print("MANUAL ROPE WALKTHROUGH")
    print("=" * 70)
    print(f"\nPositions: {positions}")
    print(f"Theta (θ): {theta}")
    print(f"\nStarting with query/key as 2D points: (x, y) = (1.0, 0.5)")
    print("\n" + "-" * 70)
    
    x0, y0 = 1.0, 0.5
    
    # Store results for later comparison
    results = []
    
    for pos in positions:
        print(f"\n📍 POSITION {pos}")
        print(f"   Step 1: Calculate angle = position / θ = {pos} / {theta}")
        angle = pos / theta
        print(f"           angle = {angle:.6f} radians ({np.degrees(angle):.2f}°)")
        
        print(f"\n   Step 2: Apply rotation matrix:")
        print(f"           [x']   [cos(θ)  -sin(θ)]   [x]")
        print(f"           [y'] = [sin(θ)   cos(θ)] × [y]")
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        print(f"\n           cos({angle:.4f}) = {cos_a:.6f}")
        print(f"           sin({angle:.4f}) = {sin_a:.6f}")
        
        x_rot, y_rot = rotate_2d(x0, y0, angle)
        
        print(f"\n   Step 3: Compute rotated coordinates:")
        print(f"           x' = x*cos(θ) - y*sin(θ)")
        print(f"              = {x0}*{cos_a:.4f} - {y0}*{sin_a:.4f}")
        print(f"              = {x_rot:.6f}")
        print(f"\n           y' = x*sin(θ) + y*cos(θ)")
        print(f"              = {x0}*{sin_a:.4f} + {y0}*{cos_a:.4f}")
        print(f"              = {y_rot:.6f}")
        
        print(f"\n   ✅ Result: ({x0:.1f}, {y0:.1f}) → ({x_rot:.6f}, {y_rot:.6f})")
        
        results.append((x_rot, y_rot))
    
    print("\n" + "=" * 70)
    print("DOT PRODUCT ANALYSIS (Query-Key Similarity)")
    print("=" * 70)
    
    # Compare all pairs
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            pos_i, pos_j = positions[i], positions[j]
            vec_i, vec_j = results[i], results[j]
            
            # Dot product
            dot_prod = vec_i[0] * vec_j[0] + vec_i[1] * vec_j[1]
            
            # Relative position
            rel_pos = abs(pos_j - pos_i)
            
            print(f"\nPosition {pos_i} ⋅ Position {pos_j}:")
            print(f"   Relative distance: {rel_pos}")
            print(f"   Dot product: {dot_prod:.6f}")
            print(f"   → {'High' if dot_prod > 0.8 else 'Medium' if dot_prod > 0.5 else 'Low'} similarity")
    
    print("\n" + "=" * 70)
    print("💡 KEY INSIGHT:")
    print("   Nearby positions (small relative distance) have higher dot products")
    print("   → They attend to each other more!")
    print("=" * 70 + "\n")
    
    return results


# =============================================================================
# VISUALIZATION 5: THETA COMPARISON
# =============================================================================

def compare_theta_values(theta_values=[10, 100, 1000, 10000], 
                        max_position=100, figsize=(15, 10)):
    """
    See what happens when you vary theta.
    This controls how far apart tokens can be before they "wrap around".
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    x0, y0 = 1.0, 0.5
    positions = np.arange(max_position)
    
    for idx, theta in enumerate(theta_values):
        ax = axes[idx]
        
        # Compute trajectory
        points = []
        for pos in positions:
            x_rot, y_rot = apply_rope_2d(x0, y0, pos, theta)
            points.append([x_rot, y_rot])
        
        points = np.array(points)
        
        # Plot with color gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))
        
        for i in range(len(positions) - 1):
            ax.plot(points[i:i+2, 0], points[i:i+2, 1], 
                   color=colors[i], linewidth=1.5, alpha=0.7)
        
        # Mark special points
        ax.scatter(points[0, 0], points[0, 1], s=150, c='green', 
                  marker='o', edgecolor='black', linewidth=2, zorder=5)
        ax.scatter(points[-1, 0], points[-1, 1], s=150, c='red', 
                  marker='s', edgecolor='black', linewidth=2, zorder=5)
        
        # Calculate total rotation
        total_angle = max_position / theta
        num_wraps = total_angle / (2 * np.pi)
        
        ax.set_title(f'θ = {theta}\n{num_wraps:.2f} full rotations', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('x dimension')
        ax.set_ylabel('y dimension')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add interpretation
        if theta < 50:
            interpretation = "⚠️ TOO FAST: wraps around quickly"
        elif theta > 5000:
            interpretation = "🐌 TOO SLOW: barely moves"
        else:
            interpretation = "✅ GOOD: smooth progression"
        
        ax.text(0.5, -0.15, interpretation, 
               transform=ax.transAxes, 
               ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('How θ affects rotation speed', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN DEMO FUNCTION
# =============================================================================

def run_full_demo():
    """
    Run all visualizations to build complete intuition about RoPE!
    """
    print("\n" + "="*70)
    print(" 🎓 WELCOME TO THE ROPE EDUCATIONAL DEMO")
    print("="*70)
    print("\nThis demo will walk you through Rotary Position Embeddings (RoPE)")
    print("step by step, with visualizations to build intuition.\n")
    
    # 1. Manual walkthrough
    print("\n📝 PART 1: Manual Walkthrough")
    print("-" * 70)
    manual_walkthrough(positions=[0, 5, 10], theta=100.0)
    input("Press Enter to continue to visualizations...")
    
    # 2. Clock analogy
    print("\n🕐 PART 2: Clock Analogy")
    print("-" * 70)
    print("Visualizing RoPE as clock hands rotating at different speeds...")
    fig1 = visualize_clock_analogy(positions=[0, 5, 10, 20], 
                                   theta_values=[10, 50, 100])
    plt.savefig("rope_clock_analogy.png")
    plt.close()
    
    # 3. 2D Spiral
    print("\n🌀 PART 3: 2D Spiral Trajectory")
    print("-" * 70)
    print("Watching how embeddings spiral as position increases...")
    fig2 = visualize_2d_spiral(max_position=50, theta=100.0)
    plt.savefig("rope_2d_spiral.png")
    plt.close()
    # 4. Attention comparison
    print("\n🎯 PART 4: Attention Pattern Comparison")
    print("-" * 70)
    print("Comparing attention with and without RoPE...")
    fig3 = visualize_attention_comparison(seq_len=20, d_model=64)
    plt.savefig("rope_attention_comparison.png")
    plt.close()
    # 5. Theta comparison
    print("\n⚙️ PART 5: Effect of Different θ Values")
    print("-" * 70)
    print("Seeing how θ controls rotation speed...")
    fig4 = compare_theta_values(theta_values=[10, 100, 1000, 10000])
    plt.savefig("rope_theta_comparison.png")
    plt.close() 
    print("\n" + "="*70)
    print(" ✅ DEMO COMPLETE!")
    print("="*70)
    print("\n💡 KEY TAKEAWAYS:")
    print("   1. RoPE rotates embeddings based on position")
    print("   2. Rotation makes nearby positions have similar angles")
    print("   3. Similar angles → high dot products → strong attention")
    print("   4. θ controls how fast rotations happen")
    print("   5. Different dimension pairs rotate at different speeds")
    print("\n" + "="*70 + "\n")


# =============================================================================
# INTERACTIVE EXPLORATION FUNCTIONS
# =============================================================================

def explore_single_pair(position=5, theta=100.0):
    """Quick visualization of a single rotation."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    x0, y0 = 1.0, 0.5
    angle = position / theta
    x_rot, y_rot = rotate_2d(x0, y0, angle)
    
    # Draw before and after
    ax.arrow(0, 0, x0, y0, head_width=0.1, head_length=0.1, 
            fc='blue', ec='blue', linewidth=3, label='Original')
    ax.arrow(0, 0, x_rot, y_rot, head_width=0.1, head_length=0.1,
            fc='red', ec='red', linewidth=3, label=f'After RoPE (pos={position})')
    
    # Draw rotation arc
    angles = np.linspace(0, angle, 50)
    radius = 0.3
    arc_x = radius * np.cos(angles + np.arctan2(y0, x0))
    arc_y = radius * np.sin(angles + np.arctan2(y0, x0))
    ax.plot(arc_x, arc_y, 'g--', linewidth=2, label=f'Rotation: {np.degrees(angle):.1f}°')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend(fontsize=12)
    ax.set_title(f'Single RoPE Rotation: pos={position}, θ={theta}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Run the full demo
    run_full_demo()
    
    # Or explore individual functions:
    # manual_walkthrough()
    # visualize_clock_analogy()
    # visualize_2d_spiral()
    # visualize_attention_comparison()
    # compare_theta_values()
    # explore_single_pair(position=10, theta=100.0)
