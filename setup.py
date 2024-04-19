from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="text_summarization_project",
    version="0.0.1",
    author="jeanettepoh",
    author_email="jeanettepoh19@gmail.com",
    description="A small python package for NLP text summarization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jeanettepoh/Text_Summarization_Project",
    project_urls={
        "Bug Tracker": f"https://github.com/jeanettepoh/Text_Summarization_Project/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src")
)