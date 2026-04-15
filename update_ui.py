import re

with open('ui/main_window.py', 'r') as f:
    content = f.read()

# 1. build_sidebar
# Change sidebar background to #080c14
content = re.sub(r'bg="#16213a"', 'bg="#080c14"', content)
# Change title text to #00ffea (the title text is previously fg="white" under bg="#16213a")
content = re.sub(r'text="FACE ATTENDANCE",\s*font=\("Segoe UI Semibold", 15\),\s*bg="#080c14",\s*fg="white"', 
                 'text="FACE ATTENDANCE",\n            font=("Segoe UI Semibold", 15),\n            bg="#080c14",\n            fg="#00ffea"', content)
# Change subtitle to #00a19e (the subtitle was fg="#a8b3c7")
content = re.sub(r'text="AI Production Edition",\s*font=\("Segoe UI", 9\),\s*bg="#080c14",\s*fg="#a8b3c7"', 
                 'text="AI Production Edition",\n            font=("Segoe UI", 9),\n            bg="#080c14",\n            fg="#00a19e"', content)

# Change navigation buttons
# bg "#05080f", fg "#a0c0ff", activebackground "#003b3b", activeforeground "#ffffff"
content = re.sub(r'bg="#1f3157",\s*fg="#eef3ff",\s*activebackground="#3a4f80",\s*activeforeground="white"',
                 'bg="#05080f",\n                fg="#a0c0ff",\n                activebackground="#003b3b",\n                activeforeground="#ffffff"', content)

# 2. build_content_area
# Change ambient_canvas and transition_canvas to bg #05080f
content = re.sub(r'self\.ambient_canvas = Canvas\(self\.content, height=56, bg="#eef3fb", highlightthickness=0\)',
                 'self.ambient_canvas = Canvas(self.content, height=56, bg="#05080f", highlightthickness=0)', content)
content = re.sub(r'self\.transition_canvas = Canvas\(self\.content, height=5, bg="#eef3fb", highlightthickness=0\)',
                 'self.transition_canvas = Canvas(self.content, height=5, bg="#05080f", highlightthickness=0)', content)

# 3. animate_active_nav_pulse
# inactive bg #05080f
content = re.sub(r'button\.config\(bg="#1f3157"\)', 'button.config(bg="#05080f")', content)
content = re.sub(r'button\.config\(bg="#2f7de1" if key == self\.active_page\.get\(\) else "#1f3157"\)',
                 'button.config(bg=self._active_nav_palette[0] if key == self.active_page.get() else "#05080f")', content)

# 4. animate_page_transition
# fill #05080f for clear, and fill #00ffea for the bar.
content = re.sub(r'canvas\.create_rectangle\(0, 0, width, 5, fill="#d8e5fb", outline=""\)',
                 'canvas.create_rectangle(0, 0, width, 5, fill="#05080f", outline="")', content)
content = re.sub(r'canvas\.create_rectangle\(position, 0, position \+ bar_width, 5, fill="#2f7de1", outline=""\)',
                 'canvas.create_rectangle(position, 0, position + bar_width, 5, fill="#00ffea", outline="")', content)

# 5. animate_ambient_glow
# ambient canvas background #05080f
content = re.sub(r'canvas\.create_rectangle\(0, 0, width, 56, fill="#eef3fb", outline=""\)',
                 'canvas.create_rectangle(0, 0, width, 56, fill="#05080f", outline="")', content)

# glowing circles to dark sci-fi tones like #081020, #0a1526, #0c1a30, #0e203b, #102545
content = re.sub(r'\["#d9e9ff", "#c7dcff", "#b7d2ff"\]', '["#081020", "#0a1526", "#0c1a30"]', content)
content = re.sub(r'\["#eaf2ff", "#ddeafe"\]', '["#0e203b", "#102545"]', content)

# 6. create_*_page functions
# change bg="white", bg="#eef3fb", bg="#d9deea" to #0c111c
content = re.sub(r'bg="white"', 'bg="#0c111c"', content)
content = re.sub(r'bg="#eef3fb"', 'bg="#0c111c"', content)
content = re.sub(r'bg="#d9deea"', 'bg="#0c111c"', content)

# fg="white", fg="#4a5470" to #00ffea or #a0c0ff
# Except in activeforeground="white", handled earlier? Wait, we replaced activeforeground="white" with "#ffffff" so it shouldn't match.
# Let's replace fg="white" with fg="#00ffea" and fg="#4a5470" with fg="#a0c0ff"
content = re.sub(r'fg="white"', 'fg="#00ffea"', content)
content = re.sub(r'fg="#4a5470"', 'fg="#a0c0ff"', content)

# 7. update_camera_frame cv2 rectangles
# blue/green to neon cyan (234, 255, 0)
# (47, 125, 225)
content = re.sub(r'\(47, 125, 225\)', '(234, 255, 0)', content)
# (0, 170, 70)
content = re.sub(r'\(0, 170, 70\)', '(234, 255, 0)', content)
# (0, 90, 220)
content = re.sub(r'\(0, 90, 220\)', '(234, 255, 0)', content)


with open('ui/main_window.py', 'w') as f:
    f.write(content)
