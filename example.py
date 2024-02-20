import dotenv
from textmancy import Processor, Annotator
from textmancy.targets import Character

dotenv.load_dotenv()

# Get text and split into pages
with open("sample_data/snows_of_kiliminjaro.txt", encoding="utf-8") as f:
    text = f.read()
    # pages = text.split("\n\n")
    paragraphs = [p for p in text.split("\n") if p]
    pages = ["".join(paragraphs[i : i + 10]) for i in range(0, len(paragraphs), 10)]

# Extract and consolidate characters
processor = Processor(target_class=Character)
results = processor.process(pages)

for r in results:
    print(r)
    print()


# Annotate pages with characters
annotater = Annotator(
    targets=results,
    model="gpt-4-1106-preview",
)
annotations = [annotater.annotate(page) for page in pages[:5]]

# Show annotations
for page, annotation in zip(pages, annotations):
    print(page)
    print(annotation)
    print([results[i].name for i in annotation])
    print()
