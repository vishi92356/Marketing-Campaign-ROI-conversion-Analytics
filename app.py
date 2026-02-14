import matplotlib.pyplot as plt
import numpy as np

# Set the general aesthetic style
plt.style.use('dark_background') # Using dark to make vibrant colors pop
fig = plt.figure(figsize=(12, 8), facecolor='white')

# Colors based on your request
LIME = '#32CD32'
LIGHT_PINK = '#FFB6C1'
VIBRANT_COLORS = ['#FF007F', '#00DFFF', '#7000FF', '#FFD700', '#00FF7F']

# --- 1. Header & Logo Placeholder ---
plt.figtext(0.5, 0.92, "BOOST YOUR BRAND", fontsize=28, fontweight='bold', 
            color=LIME, ha='center')
plt.figtext(0.5, 0.87, "MARKETING CAMPAIGN STRATEGY", fontsize=14, 
            color=LIGHT_PINK, ha='center', letterspacing=2)

# --- 2. Bar Chart (Vibrant Blue/Pink) ---
ax1 = plt.subplot(2, 2, 3)
categories = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
values = [45, 70, 55, 90, 80]
bars = ax1.bar(categories, values, color=VIBRANT_COLORS)
ax1.set_title("MONTHLY REACH", color=LIME, pad=15)
ax1.tick_params(axis='both', colors=LIGHT_PINK)

# --- 3. Line Graph (Vibrant Green) ---
ax2 = plt.subplot(2, 2, 1)
x = np.linspace(0, 10, 20)
y = np.exp(x/5)
ax2.plot(x, y, color='#00FF7F', marker='o', linewidth=3)
ax2.set_title("CONVERSION GROWTH", color=LIME, pad=15)
ax2.tick_params(axis='both', colors=LIGHT_PINK)

# --- 4. Pie Chart (Multi-color) ---
ax3 = plt.subplot(2, 2, 4)
sizes = [40, 25, 20, 15]
labels = ['Social', 'Email', 'Ads', 'Organic']
ax3.pie(sizes, labels=labels, colors=VIBRANT_COLORS, startangle=140, 
        textprops={'color': LIGHT_PINK})
ax3.set_title("TRAFFIC SOURCE", color=LIME, pad=15)

# --- 5. Custom "M" Logo (Minimalist Geometric) ---
ax4 = plt.subplot(2, 2, 2)
ax4.axis('off')
# Drawing a simple geometric 'M'
m_x = [0.2, 0.3, 0.5, 0.7, 0.8]
m_y = [0.3, 0.7, 0.5, 0.7, 0.3]
ax4.plot(m_x, m_y, color=LIGHT_PINK, linewidth=8, solid_capstyle='round')
ax4.fill(m_x, m_y, color=VIBRANT_COLORS[0], alpha=0.3)
ax4.text(0.5, 0.1, "M-AGENCY", color=LIGHT_PINK, ha='center', fontsize=12)

# Final Layout Adjustments
plt.tight_layout(rect=[0, 0.03, 1, 0.85])
plt.show()
