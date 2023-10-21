from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import uvicorn
import numpy as np
from os import environ

# Initialize FastAPI app
app = FastAPI()

# (Initial data) Define example data and categories for text classification
words = [
    "Technology: Computer, Internet, Smartphone, Software, Hardware, Code, Algorithm, Programming, Data, Artificial Intelligence (AI), Machine Learning, Virtual Reality (VR), Augmented Reality (AR), Cybersecurity, Cloud Computing, Automation, Robotics, Blockchain, 3D Printing, Nanotechnology, Internet of Things (IoT), Big Data, Wi-Fi, Social Media, Encryption, Biotechnology, Quantum Computing, GPS (Global Positioning System), Mobile Apps, Wearable Technology, Nanorobotics, Cryptocurrency, Data Science, Holography, Network Security, Quantum Cryptography, Silicon Valley, Drone Technology, Information Technology, Augmented Virtuality, Silicon Chips, Tech Startup, Internet Protocol, User Interface (UI), User Experience (UX), Data Visualization, Cyber Threats, Quantum Encryption, Tech Innovations, Machine Vision, Tech Gadgets, High-Performance Computing, Telecommunications, Firmware, Autonomous Systems, Biometric Authentication, Green Technology, Internet Infrastructure, Semiconductor Industry, Open Source Software, SQL, Database.",

    "Science: Physics, Chemistry, Biology, Astronomy, Geology, Mathematics, Ecology, Genetics, Microbiology, Meteorology, Psychology, Neuroscience, Botany, Zoology, Environmental Science, Earth Science, Quantum Mechanics, Particle Physics, Evolution, Experiment, Hypothesis, Laboratory, Research, Scientific Method, Data Analysis, Scientific Theory, Biotechnology, Climate Science, Scientific Discovery, Microscope, Telescope, Ecosystem, Fossils, Chemistry, Scientific Journal, DNA, RNA, Electromagnetism, Solar System, Geoscience, Seismology, Paleontology, Virology, Quantum Theory, Entomology, Thermodynamics, Organic Chemistry, Inorganic Chemistry, Atomic Structure, Scientific Breakthrough, Biodiversity, Theoretical Physics, Bioinformatics, Scientific Advancement, Scientific Exploration, Mathematical Modeling, Biochemistry, Research Findings, Lab Equipment, Scientific Experimentation, Ecology Conservation, Scientific Hypothesis, Chemical Reactions, Scientific Community, Analytical Chemistry, Scientific Insight, Genetics Sequencing, Climate Modeling, Science Education, Observational Astronomy, Math, Mathematics, Logic, Arithmetic, Algebra, Geometry, Trigonometry, Calculus, Statistics, Probability, Linear equations, Quadratic equations, Mathematical formulas, Mathematical notation, Mathematical symbols, Number theory, Complex numbers, Differential equations, Mathematical proofs, Mathematical modeling, Vector calculus, Mathematical concepts, Mathematical principles, Applied mathematics, Mathematical theorems, Number systems, Set theory, Mathematical functions, Mathematical axioms, Mathematical logic, Mathematical constants, Mathematical operations, Mathematical research, Modeling and simulation, Functions, Recursive functions, Celula, Bone, Enviroment, Earth, Planet",

    "Software Engineering: Coding, Programming, Development environment, Version control, Debugging, Algorithms, Testing, Agile, Scrum, Waterfall, DevOps, Continuous integration, Open source, User interface (UI), User experience (UX), Front-end, Back-end, Full-stack, Database, API (Application Programming Interface), Framework, Repository, Source code, Deployment, Software architecture, Mobile app development, Web development, Application development, Code review, Java, Python, C, C++, C#, .Net, framework, Software Lifecycle, Object-Oriented Programming (OOP), Software Testing, Software Deployment, Software Maintenance, Continuous Delivery, Scripting, Software Frameworks, Database Management, Software Prototyping, Software Requirements, Software Security, Software Scalability, Software Optimization, Software Documentation, Software Reliability, Software Performance, Software Configuration, Software Updates, Software Integration, Code Refactoring, Software Deployment Pipeline, Cross-Platform Development, Software as a Service (SaaS), Software Development Tools, Software Development Life Cycle (SDLC), Mobile Application Design, Cloud Computing for Development, Microservices, Software Development Paradigms, Data Analysis, Data, ETL, SQL, Server, Data Structures, Managment, Modeling and simulation, Functional Programming, Functions, Recursive functions, SOLID, Design Patterns, F#, Haskell, Javascript, PHP, Ruby, Scala, Swift, Kotlin, Go, Rust, R, Matlab, Assembly, Shell, Perl, Delphi, Dart, Lua, Lisp, Cobol, Fortran, Visual Basic, ActionScript, Ada, Erlang, Prolog, Clojure, Apex, ABAP, PL",

    "Business: Entrepreneurship, Marketing, Management, Finance, Strategy, Investment, Startups, Leadership, Sales, Customer Service, Innovation, Profit, Competition, Market Research, Business Plan, Corporate Culture, Sustainability, Supply Chain, Networking, E-commerce, Retail, Manufacturing, Small Business, Globalization, Risk Management, Advertising, Branding, HR (Human Resources), Taxation, Business Ethics, Business Development, Market Segmentation, Business Expansion, Economic Trends, Strategic Planning, Product Development, Market Analysis, Financial Analysis, Business Intelligence, Entrepreneurial Skills, Venture Capital, Consumer Behavior, Market Penetration, Business Growth, Supply Chain Management, Business Networking, B2B (Business-to-Business), Exporting, Economic Forecast, Business Innovation, Cost Analysis, Business Sustainability, Franchise Business, Intellectual Property, Business Leadership, Business Valuation, Marketing Strategies, Business Partnerships, Business Assets, Business Model, Youtube, Managment, ",

    "Art & Design: Painting, painting techniques, Sculpture, Drawing, Graphic Design, Photography, Architecture, Visual Arts, Illustration, Fine Arts, Fashion Design, Interior Design, Typography, Color Theory, Aesthetics, Creativity, Artistic Expression, Composition, Art Gallery, Mixed Media, Digital Art, Ceramics, Printmaking, Art History, Artistic Process, Museums, Fashion Trends, Conceptual Art, Product Design, User Experience (UX) Design, 3D Modeling, Abstraction, Collage, Installation, Sculptural, Textiles, Perspective, Craftsmanship, Pottery, Mosaic, Visual Communication, Decoupage, Woodworking, Artisanal, Artifacts, Ceramist, Textural, Craftsmanship, Futurism, Minimalism, Artisan, Fabrication, Artistry, Embroidery, Easel, Ephemera, Composition, Chiaroscuro, Printmaking, Craftsmanship, Surrealism.",

    "Teaching & Academics: Education, Teachers, Students, Classroom, Curriculum, Learning, School, College, University, Professor, Lecture, Research, Study, Exams, Homework, Pedagogy, Syllabus, Textbooks, Graduation, Tutoring, Academic Excellence, Scholarships, Academic Paper, Thesis, Degree, Classroom Management, Online Learning, E-Learning, Educational Technology, Student Engagement, Pedagogical, Academic Achievement, Educational Resources, Educational Psychology, Educational Assessment, Academic Programs, Academic Progress, Classroom Instruction, Academic Research, Educational Policy, Student Support, Educational Methods, Academic Disciplines, Academic Calendar, Academic Departments, Academic Accreditation, Academic Advising, Educational Development, Teaching Techniques, Academic Journals, Academic Conferences, Educational Philosophy, Educational Theory, Academic Standards, Academic Integrity, Classroom Environment, Educational Leadership, Academic Resources, Student Success, Academic Freedom, Managment",
    
    "Personal Development: Self-improvement, Growth, Self-awareness, Goal setting, Motivation, Mindfulness, Resilience, Empowerment, Confidence, Productivity, Time management, Positive thinking, Stress management, Emotional intelligence, Self-discipline, Well-being, Health and fitness, Leadership skills, Communication, Learning, Self-care, Life skills, Adaptability, Networking, Personal branding, Creativity, Decision-making, Problem-solving, Financial literacy, Career development, Self-mastery, Self-reflection, Personal Growth, Life Balance, Inner Strength, Self-esteem, Personal Achievement, Habits, Mindset, Self-confidence, Work-Life Balance, Positivity, Emotional Resilience, Self-control, Wellness, Physical Fitness, Interpersonal Skills, Lifelong Learning, Self-compassion, Adaptation, Relationship Building, Personal Image, Innovation, Critical Thinking, Financial Planning, Career Advancement, Youtube, Managment",

    "Health & Fitness: Exercise, Nutrition, Wellness, Cardiovascular, Strength training, Yoga, Meditation, Healthy eating, Weightlifting, Aerobics, Diet, Mental health, Physical activity, Bodyweight exercises, Calisthenics, Gym, Running, Swimming, Cycling, Flexibility, Hydration, Muscle mass, Rest and recovery, Weight management, Health goals, Personal trainer, Health tracking, Supplements, Well-being, Holistic health, Medicine, Fitness Regimen, Crossfit, Pilates, Functional Training, Body Composition, Endurance, Sports Nutrition, Mind-Body Connection, Balanced Diet, Strength Conditioning, HIIT (High-Intensity Interval Training), Wellness Retreat, Body Transformation, Outdoor Activities, Flexibility Training, Sports Psychology, Recovery Techniques, Mindful Eating, Holistic Healing, Physical Therapy, Lifestyle Medicine, Wellness Coaching, Biohacking, Wellness Programs, Active Lifestyle, Dietary Supplements, Energy Balance, Mental Resilience, Recovery Strategies, Integrative Health.",

    "Lifestyle: Habits, Choices, Routine, Culture, Preferences, Hobbies, Socializing, Leisure, Well-being, Fashion, Trends, Diet, Entertainment, Travel, Relaxation, Work-life balance, Family, Relationships, Self-care, Stress management, Personal growth, Simplicity, Mindfulness, Sustainable living, Adventure, Luxury, Minimalism, Health-consciousness, Frugality, Personal expression, Behavior, Traditions, Regimen, Lifestyle Trends, Leisure Activities, Social Engagement, Recreational Pursuits, Cultural Experiences, Well-being Practices, Fashion Styles, Cultural Diversity, Gastronomy, Cultural Exploration, Entertainment Options, Travel Destinations, Relaxation Techniques, Balance in Life, Family Dynamics, Interpersonal Bonds, Self-fulfillment, Clarity of Mind, Eco-friendly Living, Adventurous Spirit, Luxurious Experiences, Simplified Living, Mindful Awareness, Sustainable Choices, Thrilling Journeys, Opulence, Frugal Living, Youtube, Managment"
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

# Train the text classification model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(words)
clf = MultinomialNB()
clf.fit(X, categories)

# Define Pydantic model for input validation
class TextClassificationInput(BaseModel):
    title: str

# Create a POST route for text classification
@app.post("/class")
async def classify_text(item: TextClassificationInput):
    try:
        # Validate the title input
        if not item.title:
            raise HTTPException(status_code=400, detail="Title must not be empty.")
        elif len(item.title) > 64:
            raise HTTPException(status_code=400, detail="Title exceeds 64 characters.")

        # Vectorize the title
        title_vector = vectorizer.transform([item.title])

        if title_vector is None:
            raise HTTPException(status_code=500, detail="An error occurred during text vectorization.")

        # Perform text classification
        probabilities = clf.predict_proba(title_vector)[0]

        # Get available categories
        all_categories = clf.classes_

        # Get the top three categories with the highest probabilities
        top_categories = [cat for cat in all_categories if probabilities[all_categories == cat][0]]
        top_probabilities = [prob for prob in probabilities if prob]

        # Sort categories by probability in descending order
        sorted_categories = [x for _, x in sorted(zip(top_probabilities, top_categories), reverse=True)]
        sorted_probabilities = sorted(top_probabilities, reverse=True)

        # Take the top three categories
        top_3_categories = sorted_categories[:3]
        top_3_probabilities = sorted_probabilities[:3]

        # Format the response
        response = {
            "category": top_3_categories[0],
            "matches": [{"category": cat, "probability": prob} for cat, prob in zip(top_3_categories, top_3_probabilities)]
        }

        return response

    except HTTPException as e:
        return e
    except Exception as e:
        error_message = "An error occurred while processing your request."
        return HTTPException(status_code=500, detail=error_message)


# Create a GET route for get the categories
@app.get("/categories")
async def get_categories():
    try:
        return categories
    except Exception as e:
        error_message = "An error occurred while processing your request."
        return HTTPException(status_code=500, detail=error_message)


# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    port = int(environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)