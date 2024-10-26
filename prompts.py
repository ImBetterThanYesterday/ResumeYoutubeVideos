# Zero-Shot Prompt

#Este tipo de prompt no proporciona ejemplos adicionales, se basa únicamente en la instrucción y el contexto.

#Para la Traducción al Español:


translation_Template1 = """
Translate the following English summary into Spanish, using a formal tone appropriate for educational content. Ensure that all technical terms are accurately translated and that the language is clear and accessible.

Summary:
{summary}
"""
#Para el Resumen de una Transcripción de Video:


summarization_Template1 = """
Summarize the following video transcript in a clear, concise, and engaging way. Focus on the main ideas and key insights, using accessible language suitable for a general audience. Aim to create a summary that could fit within a single paragraph.

Transcript:
{transcript}

"""


#2. One-Shot Prompt
#Este prompt incluye un ejemplo adicional para guiar el modelo en su tarea.

#Para la Traducción al Español:


translation_Template2 = """
Translate the following English summary into Spanish, using a formal tone appropriate for educational content. Ensure that all technical terms are accurately translated and that the language is clear and accessible.

Example:
Summary: The principles of quantum mechanics explain the behavior of particles at the smallest scales.
Translation: Los principios de la mecánica cuántica explican el comportamiento de las partículas a las escalas más pequeñas.

Summary:
{summary}
"""
#Para el Resumen de una Transcripción de Video:


summarization_Template2 = """
Summarize the following video transcript in a clear, concise, and engaging way. Focus on the main ideas and key insights, using accessible language suitable for a general audience. Aim to create a summary that could fit within a single paragraph.

Example:
Transcript: In this video, we explore the basics of machine learning, covering topics such as supervised and unsupervised learning.
Summary: This video introduces machine learning fundamentals, including supervised and unsupervised learning techniques.

Transcript:
{transcript}
"""


#3. Few-Shot Prompt
#ste prompt incluye múltiples ejemplos, ofreciendo más contexto para la tarea.

#Para la Traducción al Español:


translation_Template3 = """
Translate the following English summary into Spanish, using a formal tone appropriate for educational content. Ensure that all technical terms are accurately translated and that the language is clear and accessible.

Examples:
Summary: The theory of relativity changed our understanding of space and time.
Translation: La teoría de la relatividad cambió nuestra comprensión del espacio y el tiempo.

Summary: Climate change is one of the most pressing issues facing humanity today.
Translation: El cambio climático es uno de los problemas más apremiantes que enfrenta a la humanidad hoy.

Summary:
{summary}
"""
#Para el Resumen de una Transcripción de Video:


summarization_Template3 = """
Summarize the following video transcript in a clear, concise, and engaging way. Focus on the main ideas and key insights, using accessible language suitable for a general audience. Aim to create a summary that could fit within a single paragraph.

Examples:
Transcript: The solar system is made up of the sun and the celestial bodies that orbit it, including planets and moons.
Summary: The solar system consists of the sun and various celestial bodies, such as planets and moons, that orbit around it.

Transcript: Artificial intelligence has the potential to revolutionize many industries, improving efficiency and productivity.
Summary: Artificial intelligence could transform various industries by enhancing efficiency and productivity.

Transcript:
{transcript}
"""