import dotenv
import logging

from textmancy.components import Processor, Annotator, PageSegmentor
from textmancy.targets import Character

dotenv.load_dotenv()

# Print logs to stream
logging.getLogger("textmancy").setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")


# Get text and split into pages
segmentor = PageSegmentor(paragraphs_per_page=10)
with open("sample_data/snows_of_kiliminjaro.txt", encoding="utf-8") as f:
    text = f.read()
    pages = segmentor.segment(text)
    # pages = pages[:5]  # Only process first 5 pages for speed

# Extract and consolidate characters
processor = Processor(target_class=Character)
results = processor.process(pages)

for r in results:
    print(r)
    print()


# Annotate pages with characters
annotater = Annotator(
    targets=results,
    model="gpt-4o",
)
annotations = [annotater.annotate(page) for page in pages[:5]]

# Show annotations
for page, annotation in zip(pages, annotations):
    print(page)
    print(annotation)
    print([results[i].name for i in annotation])
    print()
