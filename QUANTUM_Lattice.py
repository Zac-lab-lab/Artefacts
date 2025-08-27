import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set a flexible style that adapts to different sizes
plt.style.use('seaborn-v0_8')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['figure.constrained_layout.use'] = True

class VisibleLatticeVisualizer:
    def __init__(self):
        # Create figure with better proportions for visibility
        self.fig = plt.figure(figsize=(18, 14), facecolor='#f8f9fa')
        self.setup_plots()
        
    def setup_plots(self):
        """Setup the subplots with better spacing and clear layout"""
        # Create a 2x2 grid with better spacing for visibility
        gs = self.fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35, 
                                   left=0.08, right=0.95, top=0.82, bottom=0.18)
        
        # Main 3D lattice visualization
        self.ax1 = self.fig.add_subplot(gs[0, 0], projection='3d')
        self.ax1.set_title('🔷 LWE Lattice Structure\n(Basis Vectors)', 
                           fontsize=14, fontweight='bold', color='#2c3e50', pad=25)
        
        # Secret vector in lattice
        self.ax2 = self.fig.add_subplot(gs[0, 1], projection='3d')
        self.ax2.set_title('🔐 Secret Vector s\nin Lattice Space', 
                           fontsize=14, fontweight='bold', color='#2c3e50', pad=25)
        
        # Encryption process
        self.ax3 = self.fig.add_subplot(gs[1, 0])
        self.ax3.set_title('🔒 LWE Encryption Process\n(b = A·s + e)', 
                           fontsize=14, fontweight='bold', color='#2c3e50', pad=25)
        
        # Decryption process
        self.ax4 = self.fig.add_subplot(gs[1, 1])
        self.ax4.set_title('🔓 Decryption Decision\n(Closest Vector)', 
                           fontsize=14, fontweight='bold', color='#2c3e50', pad=25)
        
        # Set consistent background colors and styling for all axes
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            if hasattr(ax, 'set_facecolor'):
                ax.set_facecolor('#ffffff')
            # Set consistent grid styling with better visibility
            if hasattr(ax, 'grid'):
                ax.grid(True, alpha=0.3, color='#95a5a6', linewidth=1.2)
        
    def create_lattice_basis(self):
        """Create a simple 3D lattice basis"""
        # Create orthogonal basis vectors for clear visualization
        basis = np.array([
            [1, 0, 0],  # b1
            [0, 1, 0],  # b2  
            [0, 0, 1]   # b3
        ])
        return basis
    
    def generate_lattice_points(self, basis, range_val=2):
        """Generate lattice points from basis vectors"""
        points = []
        for i in range(-range_val, range_val + 1):
            for j in range(-range_val, range_val + 1):
                for k in range(-range_val, range_val + 1):
                    point = i * basis[0] + j * basis[1] + k * basis[2]
                    points.append(point)
        return np.array(points)
    
    def plot_lattice_structure(self):
        """Plot the main 3D lattice structure with high visibility"""
        basis = self.create_lattice_basis()
        points = self.generate_lattice_points(basis, 2)
        
        # Plot lattice points with high contrast colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(points)))
        self.ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                         c=colors, alpha=0.9, s=80, edgecolors='black', linewidth=1)
        
        # Plot basis vectors with high visibility styling
        origin = np.zeros(3)
        colors = ['#e74c3c', '#27ae60', '#3498db']  # Red, Green, Blue
        labels = ['b₁ (Basis Vector 1)', 'b₂ (Basis Vector 2)', 'b₃ (Basis Vector 3)']
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            self.ax1.quiver(origin[0], origin[1], origin[2],
                           basis[i, 0], basis[i, 1], basis[i, 2],
                           color=color, arrow_length_ratio=0.25, linewidth=6,
                           label=label, alpha=1.0)
        
        # Add connecting lines for structure with high visibility
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if abs(i) + abs(j) + abs(k) <= 1:
                        point = i * basis[0] + j * basis[1] + k * basis[2]
                        # Draw lines to neighboring points
                        for di, dj, dk in [(1,0,0), (0,1,0), (0,0,1)]:
                            if abs(i+di) <= 1 and abs(j+dj) <= 1 and abs(k+dk) <= 1:
                                neighbor = (i+di) * basis[0] + (j+dj) * basis[1] + (k+dk) * basis[2]
                                self.ax1.plot([point[0], neighbor[0]], 
                                            [point[1], neighbor[1]], 
                                            [point[2], neighbor[2]], 
                                            color='#34495e', alpha=0.8, linewidth=2.5)
        
        # Clear axis labels with descriptions
        self.ax1.set_xlabel('X Axis\n(Lattice Dimension 1)', fontweight='bold', color='#2c3e50', fontsize=12)
        self.ax1.set_ylabel('Y Axis\n(Lattice Dimension 2)', fontweight='bold', color='#2c3e50', fontsize=12)
        self.ax1.set_zlabel('Z Axis\n(Lattice Dimension 3)', fontweight='bold', color='#2c3e50', fontsize=12)
        
        # Clear legend with better positioning and visibility
        legend = self.ax1.legend(frameon=True, fancybox=True, shadow=True, 
                                loc='upper left', fontsize=11, bbox_to_anchor=(0, 1))
        legend.get_frame().set_facecolor('#ffffff')
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_edgecolor('#2c3e50')
        legend.get_frame().set_linewidth(1.5)
        
        self.ax1.view_init(elev=25, azim=45)
        
        # Set clear axis limits
        self.ax1.set_xlim(-2.5, 2.5)
        self.ax1.set_ylim(-2.5, 2.5)
        self.ax1.set_zlim(-2.5, 2.5)
        
        # Add clear tick labels
        self.ax1.set_xticks([-2, -1, 0, 1, 2])
        self.ax1.set_yticks([-2, -1, 0, 1, 2])
        self.ax1.set_zticks([-2, -1, 0, 1, 2])
        
    def plot_secret_vector(self):
        """Plot the secret vector in 3D lattice space with high visibility"""
        basis = self.create_lattice_basis()
        
        # Define secret vector s = [3, 5, 2] in 3D
        s = np.array([3, 5, 2])
        
        # Plot basis vectors with clear colors
        origin = np.zeros(3)
        colors = ['#e74c3c', '#27ae60', '#3498db']
        
        for i, color in enumerate(colors):
            self.ax2.quiver(origin[0], origin[1], origin[2],
                           basis[i, 0], basis[i, 1], basis[i, 2],
                           color=color, alpha=0.8, arrow_length_ratio=0.25, linewidth=4)
        
        # Plot secret vector with high visibility styling
        secret_point = basis.T @ s
        self.ax2.quiver(origin[0], origin[1], origin[2],
                        secret_point[0], secret_point[1], secret_point[2],
                        color='#9b59b6', arrow_length_ratio=0.25, linewidth=8,
                        label=f'Secret Vector s = [{s[0]}, {s[1]}, {s[2]}]\n(Private Key)', alpha=1.0)
        
        # Add lattice points for context with high contrast
        points = self.generate_lattice_points(basis, 1)
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(points)))
        self.ax2.scatter(points[:, 0], points[:, 1], points[:, 2], 
                         c=colors, alpha=0.8, s=60, edgecolors='black', linewidth=1)
        
        # Clear axis labels with descriptions
        self.ax2.set_xlabel('X Axis\n(Lattice Dimension 1)', fontweight='bold', color='#2c3e50', fontsize=12)
        self.ax2.set_ylabel('Y Axis\n(Lattice Dimension 2)', fontweight='bold', color='#2c3e50', fontsize=12)
        self.ax2.set_zlabel('Z Axis\n(Lattice Dimension 3)', fontweight='bold', color='#2c3e50', fontsize=12)
        
        # Clear legend with better positioning and visibility
        legend = self.ax2.legend(frameon=True, fancybox=True, shadow=True, 
                                loc='upper left', fontsize=11, bbox_to_anchor=(0, 1))
        legend.get_frame().set_facecolor('#ffffff')
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_edgecolor('#2c3e50')
        legend.get_frame().set_linewidth(1.5)
        
        self.ax2.view_init(elev=25, azim=45)
        
        # Set clear axis limits
        self.ax2.set_xlim(-1.5, 1.5)
        self.ax2.set_ylim(-1.5, 1.5)
        self.ax2.set_zlim(-1.5, 1.5)
        
        # Add clear tick labels
        self.ax2.set_xticks([-1, 0, 1])
        self.ax2.set_yticks([-1, 0, 1])
        self.ax2.set_zticks([-1, 0, 1])
        
    def plot_encryption_process(self):
        """Plot the LWE encryption process with high visibility"""
        # Define the matrices from the example
        A = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        s = np.array([3, 5])
        e = np.array([0, 1, -1, 0])
        
        # Calculate b = A*s + e (mod 17)
        b_no_error = A @ s
        b = (b_no_error + e) % 17
        
        # Create visualization with high contrast colors and spacing
        x_pos = np.arange(len(b))
        width = 0.4
        
        # Create high contrast gradient colors for bars
        colors1 = plt.cm.Blues(np.linspace(0.3, 0.9, len(b_no_error)))
        colors2 = plt.cm.Oranges(np.linspace(0.3, 0.9, len(b)))
        
        # Plot A*s with high contrast gradient
        bars1 = self.ax3.bar(x_pos - width/2, b_no_error, width, 
                             label='A·s (Matrix Multiplication)', color=colors1, alpha=0.95, 
                             edgecolor='black', linewidth=1.5)
        
        # Plot b (with error) with high contrast gradient
        bars2 = self.ax3.bar(x_pos + width/2, b, width, 
                             label='b = A·s + e (mod 17)\n(With Error)', color=colors2, alpha=0.95,
                             edgecolor='black', linewidth=1.5)
        
        # Add clear error annotations with high visibility
        for i, (no_err, with_err, err_val) in enumerate(zip(b_no_error, b, e)):
            if err_val != 0:
                self.ax3.annotate(f'Error e={err_val}', 
                                 xy=(i, max(no_err, with_err)), 
                                 xytext=(i, max(no_err, with_err) + 6),
                                 arrowprops=dict(arrowstyle='->', color='#e74c3c', 
                                               lw=3, alpha=0.9),
                                 ha='center', fontsize=11, fontweight='bold',
                                 bbox=dict(boxstyle="round,pad=0.4", 
                                          facecolor='#f39c12', alpha=0.9,
                                          edgecolor='#e67e22', linewidth=2))
        
        # Clear axis labels with descriptions
        self.ax3.set_xlabel('Vector Components\n(Matrix Rows)', fontweight='bold', color='#2c3e50', fontsize=12)
        self.ax3.set_ylabel('Values (mod 17)\n(Modular Arithmetic)', fontweight='bold', color='#2c3e50', fontsize=12)
        
        # Clear legend with better positioning and visibility
        legend = self.ax3.legend(frameon=True, fancybox=True, shadow=True, 
                                loc='upper right', fontsize=11)
        legend.get_frame().set_facecolor('#ffffff')
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_edgecolor('#2c3e50')
        legend.get_frame().set_linewidth(1.5)
        
        # Add clear value labels on bars with high visibility
        for bar in bars1:
            height = bar.get_height()
            self.ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                         f'{int(height)}', ha='center', va='bottom', 
                         fontweight='bold', fontsize=11, color='#2c3e50')
        
        for bar in bars2:
            height = bar.get_height()
            self.ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                         f'{int(height)}', ha='center', va='bottom', 
                         fontweight='bold', fontsize=11, color='#2c3e50')
        
        # Set clear axis limits
        self.ax3.set_ylim(0, max(b_no_error) + 12)
        
        # Add clear tick labels
        self.ax3.set_xticks(x_pos)
        self.ax3.set_xticklabels([f'Row {i+1}' for i in range(len(b))])
        
    def plot_decryption_process(self):
        """Plot the decryption decision process with high visibility"""
        # Values from the example
        q = 17
        floor_q_2 = q // 2  # 8
        
        # Simulate decryption values
        np.random.seed(42)
        n_samples = 60
        v_values = np.random.uniform(0, q, n_samples)
        
        # Categorize as 0 or 1 based on distance
        decisions = []
        for v in v_values:
            dist_to_0 = min(v, q - v)
            dist_to_8 = min(abs(v - floor_q_2), q - abs(v - floor_q_2))
            decisions.append(0 if dist_to_0 < dist_to_8 else 1)
        
        # Create high visibility scatter plot with clear colors
        colors = ['#3498db' if d == 0 else '#e74c3c' for d in decisions]  # Blue for 0, Red for 1
        self.ax4.scatter(v_values, [0] * len(v_values), c=colors, alpha=0.9, s=100, 
                         edgecolors='black', linewidth=1.5)
        
        # Add clear decision boundaries with high visibility
        self.ax4.axvline(x=0, color='#3498db', linestyle='--', alpha=0.9, 
                         linewidth=4, label='m=0 region\n(Decrypt to 0)')
        self.ax4.axvline(x=floor_q_2, color='#e74c3c', linestyle='--', alpha=0.9, 
                         linewidth=4, label='m=1 region\n(Decrypt to 1)')
        self.ax4.axvline(x=q, color='#3498db', linestyle='--', alpha=0.9, linewidth=3)
        
        # Add clear example point with high visibility
        example_v = 7
        self.ax4.scatter(example_v, 0, color='#27ae60', s=400, marker='*', 
                         label=f'Example: v={example_v} → m=1\n(Correct Decryption)', 
                         edgecolors='black', linewidth=2)
        
        # Clear axis labels with descriptions
        self.ax4.set_xlabel('Decryption Value v\n(Closest Vector Distance)', fontweight='bold', color='#2c3e50', fontsize=12)
        self.ax4.set_ylabel('')
        
        # Clear legend with better positioning and visibility
        legend = self.ax4.legend(frameon=True, fancybox=True, shadow=True, 
                                loc='upper right', fontsize=11)
        legend.get_frame().set_facecolor('#ffffff')
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_edgecolor('#2c3e50')
        legend.get_frame().set_linewidth(1.5)
        
        self.ax4.set_ylim(-0.6, 0.6)
        
        # Add clear region labels with high visibility
        self.ax4.text(q/4, 0.4, 'm=0\nRegion\n(Decrypt to 0)', ha='center', va='center', 
                      fontweight='bold', color='#3498db', fontsize=12,
                      bbox=dict(boxstyle="round,pad=0.4", facecolor='#ebf3fd', 
                               alpha=0.95, edgecolor='#3498db', linewidth=2))
        self.ax4.text(3*q/4, 0.4, 'm=1\nRegion\n(Decrypt to 1)', ha='center', va='center', 
                      fontweight='bold', color='#e74c3c', fontsize=12,
                      bbox=dict(boxstyle="round,pad=0.4", facecolor='#fdf2f2', 
                               alpha=0.95, edgecolor='#e74c3c', linewidth=2))
        
        # Set clear axis limits
        self.ax4.set_xlim(-1, q + 1)
        
        # Add clear tick labels
        self.ax4.set_xticks([0, 4, 8, 12, 16])
        self.ax4.set_xticklabels(['0', '4', '8\n(m=1)', '12', '16'])
        
    def add_info_text(self):
        """Add clear and well-formatted information text with high visibility"""
        info_text = """
        🔐 LWE (Learning With Errors) Cryptography
        
        • Lattice: Discrete subgroup of Rⁿ (mathematical foundation)
        • Security: Based on SVP/CVP hardness (computationally hard)
        • Post-quantum: Resistant to Shor's algorithm (quantum-safe)
        • Key: Secret vector s hidden by noise (private key)
        • Encryption: b = A·s + e (mod q) (adds noise)
        • Decryption: Find closest lattice point (error correction)
        """
        
        self.fig.text(0.02, 0.02, info_text, fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=1.0", facecolor="#e8f5e8", 
                              edgecolor="#27ae60", alpha=0.95, linewidth=2),
                     color='#2c3e50')
    
    def visualize(self):
        """Create the complete high visibility visualization"""
        print("🎨 Creating High Visibility 3D Lattice Visualization...")
        
        # Create all plots
        self.plot_lattice_structure()
        self.plot_secret_vector()
        self.plot_encryption_process()
        self.plot_decryption_process()
        
        # Add information text
        self.add_info_text()
        
        # Add clear main title with proper spacing to prevent overlapping
        self.fig.suptitle('🔷 LWE (Learning With Errors) Lattice Cryptography\n'
                         '🎯 3D Visualization of Post-Quantum Security', 
                         fontsize=18, fontweight='bold', color='#2c3e50', y=0.94)
        
        # Add clear subtitle with proper spacing
        self.fig.text(0.5, 0.89, 'Quantum-Resistant Cryptography Based on Lattice Problems', 
                     ha='center', fontsize=13, style='italic', color='#7f8c8d')
        
        # Use constrained layout for better flexibility
        self.fig.set_constrained_layout(True)
        plt.show()
        
        print("✨ High visibility visualization complete!")

def demonstrate_lwe_example():
    """Demonstrate the specific LWE example with clear formatting"""
    print("\n" + "="*60)
    print("🔢 LWE Example Demonstration")
    print("="*60)
    
    # Parameters
    q = 17
    s = np.array([3, 5])  # Secret vector
    A = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])  # Public matrix
    e = np.array([0, 1, -1, 0])  # Error vector
    
    print(f"📊 Parameters:")
    print(f"   Modulus q = {q}")
    print(f"   Secret vector s = {s}")
    print(f"   Error vector e = {e}")
    
    # Key generation
    b_no_error = A @ s
    b = (b_no_error + e) % q
    
    print(f"\n🔑 Key Generation:")
    print(f"   A·s = {b_no_error}")
    print(f"   b = A·s + e (mod {q}) = {b}")
    
    # Encryption
    r = np.array([1, 0, 1, 0])  # Random vector
    m = 1  # Message bit
    
    c1 = (A.T @ r) % q
    c2 = (b @ r + m * (q // 2)) % q
    
    print(f"\n🔒 Encryption (m = {m}):")
    print(f"   Random vector r = {r}")
    print(f"   c₁ = A^T·r (mod {q}) = {c1}")
    print(f"   c₂ = b^T·r + m·⌊q/2⌋ (mod {q}) = {c2}")
    
    # Decryption
    v = (c2 - c1 @ s) % q
    distance_to_0 = min(v, q - v)
    distance_to_8 = min(abs(v - 8), q - abs(v - 8))
    decrypted = 1 if distance_to_8 < distance_to_0 else 0
    
    print(f"\n🔓 Decryption:")
    print(f"   v = c₂ - ⟨c₁, s⟩ (mod {q}) = {v}")
    print(f"   Distance to 0: {distance_to_0}")
    print(f"   Distance to 8: {distance_to_8}")
    print(f"   Decrypted message: {decrypted} {'✅' if decrypted == m else '❌'}")
    
    print(f"\n📈 Security Analysis:")
    print(f"   • Lattice dimension: {A.shape[1]}")
    print(f"   • Number of samples: {A.shape[0]}")
    print(f"   • Error bound: max(|e|) = {np.max(np.abs(e))}")
    print(f"   • Noise term: r^T·e = {r @ e}")

if __name__ == "__main__":
    # Create the high visibility visualization
    visualizer = VisibleLatticeVisualizer()
    visualizer.visualize()
    
    # Demonstrate the LWE example
    demonstrate_lwe_example()
    
    print("\n" + "="*60)
    print("🎯 High Visibility 3D Lattice Visualization Complete!")
    print("="*60)
    print("✨ Features:")
    print("   • 3D lattice structure with basis vectors")
    print("   • Secret vector in lattice space")
    print("   • LWE encryption process with errors")
    print("   • Decryption decision process")
    print("   • Post-quantum security demonstration")
    print("   • High visibility and clear information")
