SubGraphPrompt = """Bug Report is provided with Bug Scenario Images Description. Please analyze the bug scenario images to infer possible bug root cause.
The image description has the following formats:
1. HTML format: If the Bug Scenario Images is a screenshot of a frontend webpage error
2. UML code/Mermaid: If the Bug Scenario Images is a flowchart/sequence diagram, etc.
3. Natural language: If the Bug Scenario Images is a natural image

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits (i.e., patches) to fix the issue. Use the suggested solution from the Bug Report directly to fix the bug.

INPUT: 

* Bug Report
'''
{{problem_statement}}
'''

* Bug Scenario Images
'''
{{image_captioning}}
'''

* Below is the description of the most relevant part of the image related to the bug. Please focus on this section to locate and fix the problematic code.
'''
{{sub_image_caption}}
'''


Use the suggested solution from the Bug Report directly to fix the bug!
YOU MUST Conduct a careful analysis to ensure your patch is both executable and effectively resolves the stated problem!！
"""



ImagePrompt = """Bug Report is provided with Bug Scenario Images Description. Please analyze the bug scenario images to infer possible bug root cause.
The image description has the following formats:
1. HTML format: If the Bug Scenario Images is a screenshot of a frontend webpage error
2. UML code/Mermaid: If the Bug Scenario Images is a flowchart/sequence diagram, etc.
3. Natural language: If the Bug Scenario Images is a natural image

Please first localize the bug based on the issue statement, and output 与这个bug相关的代码片段

INPUT: 

* Bug Report
'''
{{problem_statement}}
'''

* Bug Scenario Images
'''
{{image_captioning}}
'''

YOU MUST Conduct a careful analysis to ensure your localization result is correct!！
"""


VLMPrompt = """You are a master at analyzing images and code.

# Task  
I will provide you with a bug report, an image related to the bug (image resolution={{resolution}}), and possibly the code snippet(s) that correspond to the bug. Your job is to analyze both the bug description and the code snippet(s), locate the region in the image that is most relevant to this bug, and return its bounding-box coordinates.

# Input  

* Bug Report  
'''
{{problem_statement}}
'''

* Code snippets  
'''
{{code_snips}}
'''

# Output format  
<reason>  
Please describe your reasoning process.  
</reason>  
<result>Return the coordinates of the relevant region in the form [x, y, w, h], where x and y are the coordinates of the top-left corner of the bounding box, and w and h are its width and height.</result>"""