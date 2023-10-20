from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import uvicorn
from os import environ

app = FastAPI()

# initial example data
words = [
    "Technology: Computer, Internet, Smartphone, Software, Hardware, Code, Algorithm, Programming, Data, Artificial Intelligence (AI), Machine Learning, Virtual Reality (VR), Augmented Reality (AR), Cybersecurity, Cloud Computing, Automation, Robotics, Blockchain, 3D Printing, Nanotechnology, Internet of Things (IoT), Big Data, Wi-Fi, Social Media, Encryption, Biotechnology, Quantum Computing, GPS (Global Positioning System), Mobile Apps, Wearable Technology.",

    "Science: Physics, Chemistry, Biology, Astronomy, Geology, Mathematics, Ecology, Genetics, Microbiology, Meteorology, Psychology, Neuroscience, Botany, Zoology, Environmental Science, Earth Science, Quantum Mechanics, Particle Physics, Evolution, Experiment, Hypothesis, Laboratory, Research, Scientific Method, Data Analysis, Scientific Theory, Biotechnology, Climate Science, Scientific Discovery, Microscope, Telescope, Ecosystem, Fossils, Chemistry, Scientific Journal, DNA, RNA, Electromagnetism, Solar System, Geoscience.",

    "Software Development: Coding, Programming, Software engineering, Development environment, Version control, Debugging, Algorithms, Testing, Agile, Scrum, Waterfall, DevOps, Continuous integration, Open source, User interface (UI), User experience (UX), Front-end, Back-end, Full-stack, Database, API (Application Programming Interface), Framework, Repository, Source code, Deployment, Software architecture, Mobile app development, Web development, Application development, Code review, Java, Python, C, C++, C#, .Net, framework",

    "Business: Entrepreneurship, Marketing, Management, Finance, Strategy, Investment, Startups, Leadership, Sales, Customer Service, Innovation, Profit, Competition, Market Research, Business Plan, Corporate Culture, Sustainability, Supply Chain, Networking, E-commerce, Retail, Manufacturing, Small Business, Globalization, Risk Management, Advertising, Branding, HR (Human Resources), Taxation, Business Ethics.",

    "Art & Design: Painting, Sculpture, Drawing, Graphic Design, Photography, Architecture, Visual Arts, Illustration, Fine Arts, Fashion Design, Interior Design, Typography, Color Theory, Aesthetics, Creativity, Artistic Expression, Composition, Art Gallery, Mixed Media, Digital Art, Ceramics, Printmaking, Art History, Artistic Process, Museums, Fashion Trends, Conceptual Art, Product Design, User Experience (UX) Design, 3D Modeling.",

    "Teaching & Academics: Education, Teachers, Students, Classroom, Curriculum, Learning, School, College, University, Professor, Lecture, Research, Study, Exams, Homework, Pedagogy, Syllabus, Textbooks, Graduation, Tutoring, Academic Excellence, Scholarships, Academic Paper, Thesis, Degree, Classroom Management, Online Learning, E-Learning, Educational Technology, Student Engagement.",
    
    "Personal Development: Self-improvement, Growth, Self-awareness, Goal setting, Motivation, Mindfulness, Resilience, Empowerment, Confidence, Productivity, Time management, Positive thinking, Stress management, Emotional intelligence, Self-discipline, Well-being, Health and fitness, Leadership skills, Communication, Learning, Self-care, Life skills, Adaptability, Networking, Personal branding, Creativity, Decision-making, Problem-solving, Financial literacy, Career development.",

    "Health & Fitness: Exercise, Nutrition, Wellness, Cardiovascular, Strength training, Yoga, Meditation, Healthy eating, Weightlifting, Aerobics, Diet, Mental health, Physical activity, Bodyweight exercises, Calisthenics, Gym, Running, Swimming, Cycling, Flexibility, Hydration, Muscle mass, Rest and recovery, Weight management, Health goals, Personal trainer, Health tracking, Supplements, Well-being, Holistic health.",

    "Habits, Choices, Routine, Culture, Preferences, Hobbies, Socializing, Leisure, Well-being, Fashion, Trends, Diet, Entertainment, Travel, Relaxation, Work-life balance, Family, Relationships, Self-care, Stress management, Personal growth, Simplicity, Mindfulness, Sustainable living, Adventure, Luxury, Minimalism, Health-consciousness, Frugality, Personal expression."
]

categories = [
    "Technology",
    "Science",
    "Software Development",
    "Business",
    "Art & Design",
    "Teaching & Academics",
    "Personal Development",
    "Health & Fitness",
    "Lifestyle"
]

# Entrenamiento del modelo de clasificación
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(words)
clf = MultinomialNB()
clf.fit(X, categories)

class TextClassificationInput(BaseModel):
    title: str

@app.post("/class")
async def classify_text(item: TextClassificationInput):
    """
    Classify text into a category.
    :param item: title of the note.
    :return: Predicted category.
    """

    title = item.title

    if not title:
        return {"error": "Title must not be empty."}
    elif len(title) > 64:
        return {"error": "Title exceeds 64 characters."}

    title_vector = vectorizer.transform([title])
    category = clf.predict(title_vector)[0]
    
    return {"category": category}


if __name__ == "__main__":
    port = int(environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)