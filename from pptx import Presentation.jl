from pptx import Presentation
from pptx.util import Inches, Pt

# Create a new presentation
prs = Presentation()

# Function to create slide with icons and text
def add_icon_text_slide(prs, title, icons_text):
    layout = prs.slide_layouts[5]  # Title Only layout
    slide = prs.slides.add_slide(layout)
    shapes = slide.shapes
    title_shape = shapes.title
    title_shape.text = title

    left = Inches(1)
    top = Inches(1.5)
    width = Inches(8)
    height = Inches(5)
    textbox = shapes.add_textbox(left, top, width, height)
    tf = textbox.text_frame

    # Add icons and descriptions
    for icon, subtitle, description in icons_text:
        p = tf.add_paragraph()
        p.text = f"{icon} {subtitle}\n{description}"
        p.font.size = Pt(20)
        p.level = 0

# AI Weaknesses
weaknesses = [
    ("🔒", "Data Privacy and Security", "Risk of personal data breaches."),
    ("💸", "Cost and Implementation Challenges", "Hard for small organizations to afford."),
    ("🤖", "Over-Reliance on Technology", "Weakens human judgment and response.")
]
add_icon_text_slide(prs, "AI Weaknesses", weaknesses)

# AI Opportunities
opportunities = [
    ("🚀", "Creating Safer Work Environments", "New AI tools can monitor and prevent incidents."),
    ("❤️", "Improving Worker Well-Being", "Health monitoring and stress detection tools."),
    ("🐷", "Reducing Costs in the Long Run", "Fewer accidents and insurance claims."),
    ("💡", "Developing New OSH Tools and Technologies", "Innovation in safety wearables, apps, and systems."),
    ("📈", "Increasing Efficiency", "Faster and better decision-making.")
]
add_icon_text_slide(prs, "AI Opportunities", opportunities)

# AI Threats
threats = [
    ("🚶", "Job Displacement", "Some safety roles may be automated."),
    ("🛡️", "Cybersecurity Threats", "AI systems can be hacked, risking safety."),
    ("🙅", "Resistance to Change", "Workers or management may resist AI adoption."),
    ("⚖️", "Lack of Regulatory Frameworks", "No strong global standards yet for AI safety in workplaces.")
]
add_icon_text_slide(prs, "AI Threats", threats)

# Add SWOT Quadrant slide
slide_layout = prs.slide_layouts[5]
slide = prs.slides.add_slide(slide_layout)
shapes = slide.shapes
title_shape = shapes.title
title_shape.text = "SWOT Analysis of AI in OSH"
left = Inches(0.5)
top = Inches(1.5)
width = Inches(9)
height = Inches(5)

textbox = shapes.add_textbox(left, top, width, height)
tf = textbox.text_frame
p = tf.add_paragraph()
p.text = (
    "Strengths                   | Weaknesses\n"
    "-----------------------------------------------\n"
    "- Enhances Safety           | - Data Privacy & Security\n"
    "- Improved Decision-Making  | - Cost & Implementation Challenges\n"
    "- Risk Management           | - Over-Reliance on Technology\n"
    "- Increased Efficiency      | \n"
    "- Information on Hazardous Tasks| \n\n"
    "Opportunities               | Threats\n"
    "-----------------------------------------------\n"
    "- Safer Work Environments   | - Job Displacement\n"
    "- Worker Well-Being         | - Resistance to Change\n"
    "- Reducing Costs            | - Cybersecurity Threats\n"
    "- Increasing Efficiency     | - Lack of Regulatory Frameworks\n"
    "- New OSH Tools & Tech      | "
)
p.font.size = Pt(18)

# Save the presentation
pptx_file = '/mnt/data/AI_OSH_SWOT_Presentation.pptx'
prs.save(pptx_file)

pptx_file
