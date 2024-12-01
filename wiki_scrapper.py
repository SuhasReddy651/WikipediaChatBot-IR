import warnings
from bs4 import GuessedAtParserWarning
import wikipedia
import pandas as pd
import re
import json

warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

class WikipediaScraper:
    def __init__(self):
        self.known_titles = set()
        self.known_rev_id = set()
        self.documents = []
    
    def scrape_wikipedia(self, query, num_search_results=500, category=None):
        search_results = wikipedia.search(query, results=num_search_results * 10)
        for result in search_results:
            if len(self.documents) >= num_search_results:
                break

            try:
                page = wikipedia.page(result, auto_suggest=False)

                if page.title in self.known_titles or page.revision_id in self.known_rev_id:
                    continue
                if len(page.summary) < 200:
                    continue

                self.documents.append({
                    'title': page.title,
                    'revision_id': page.revision_id,
                    'summary': page.summary,
                    'url': page.url,
                    'category': category
                })

                self.known_titles.add(page.title)
                self.known_rev_id.add(page.revision_id)

            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
                continue

    def process_topics(self, topics, output_csv='scraped_dt.csv', output_json='scraped_data.json'):
        for category, subtopics in topics.items():
            for topic in subtopics:
                print(f"Scraping Wikipedia for: {topic}")
                self.scrape_wikipedia(topic, category=category)
                print(f"Retrieved {len(self.documents)} pages for the topic '{topic}' in category '{category}'")
        
        df = pd.DataFrame(self.documents)
        df['summary_length'] = df['summary'].apply(len)
        df = df[df['summary_length'] > 200]
        df['cleaned_summary'] = df['summary'].apply(self.stopwords_removal)
        df = df.drop(columns=['summary', 'summary_length'])
        df = df.rename(columns={'cleaned_summary': 'summary', 'category': 'topic'})

        # Save to CSV
        df.to_csv(output_csv, index=False)
        print(f"Data saved to '{output_csv}'")

        # Convert to JSON
        grouped_data = {}
        for category, group in df.groupby('topic'):
            grouped_data[category] = group.to_dict(orient='records')
        
        with open(output_json, 'w') as json_file:
            json.dump(grouped_data, json_file, indent=4)
            print(f"Converted dataframe to JSON format: {output_json}")

    @staticmethod
    def stopwords_removal(text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)


if __name__ == "__main__":
    topics = {
    'Health': [
        "Common diseases", "Global health statistics", "Mental health trends",
        "Nutrition and wellness", "Preventive healthcare", "Vaccine development and distribution",
        "Healthcare access and inequality", "Impact of technology on healthcare", "Women's health issues", "Global pandemics and responses", "Chronic disease management", "Rare diseases",
        "Healthcare system reforms", "Telemedicine advancements", "Aging population challenges"
    ],
    'Environment': [
        "Global warming", "Endangered species", "Deforestation rates",
        "Renewable energy adoption", "Plastic pollution and recycling", "Ocean acidification",
        "Air quality and pollution", "Biodiversity loss", "Sustainable development goals (SDGs)",
        "Impact of urbanization on the environment", "Climate adaptation strategies", "Wildlife conservation",
        "Eco-friendly innovations", "Carbon capture technologies", "Global environmental policies"
    ],
    'Technology': [
        "Emerging technologies", "AI advancements", "Robotics",
        "Cybersecurity trends", "Blockchain and Web3 applications", "Internet of Things (IoT)",
        "Quantum computing developments", "Virtual and augmented reality", "5G and future network technologies",
        "Ethical issues in technology", "Autonomous vehicles", "Wearable technology",
        "Space exploration technologies", "Green tech innovations", "Digital transformation in industries",
        "Open-source software trends", "Cloud computing advancements", "Digital privacy concerns",
        "Big data analytics", "Human-computer interaction"
    ],
    'Economy': [
        "Stock market performance", "Job markets", "Cryptocurrency trends",
        "Global trade and tariffs", "Economic impacts of climate change", "Recession indicators and trends",
        "Small business and entrepreneurship", "E-commerce growth", "Real estate market analysis",
        "Government stimulus and fiscal policies", "Consumer spending patterns", "Global debt concerns",
        "Wealth inequality", "Tax reforms", "Inflation and deflation trends",
        "International financial organizations", "Labor market automation", "Green economy initiatives",
        "Foreign direct investment", "Startup ecosystems"
    ],
    'Sports': [
        "Football", "Cricket", "Badminton",
        "Olympic sports and updates", "Women's sports leagues", "eSports and virtual gaming competitions",
        "Doping scandals and sports ethics", "Fan engagement and digital innovation in sports", "Major international tournaments",
        "Rising sports trends", "Sports science and athlete recovery", "Youth sports development",
        "Paralympic sports", "Coaching and leadership in sports", "Sports marketing and sponsorships",
        "Sports betting and regulations", "Globalization of sports leagues", "Historic sports moments",
        "Athlete activism", "Grassroots sports programs"
    ],
    'Entertainment': [
        "Music industry", "Popular cultural events", "Streaming platforms",
        "Film and television industry trends", "Gaming and eSports", "Celebrity culture and influence",
        "Independent and alternative art movements", "Awards shows and red-carpet events", "Impact of AI on creative arts",
        "Globalization of entertainment content", "Rise of podcasts", "Virtual concerts and events",
        "Fan-driven content creation", "Revivals and reboots in media", "Influencer marketing in entertainment",
        "Crossover content trends", "Cultural preservation through media", "Historical documentaries",
        "Music festivals and tours", "Technological innovation in the arts"
    ],
    'Politics': [
        "Elections", "Public policy analysis", "International relations",
        "Political ideologies and movements", "Role of social media in politics", "Human rights and democracy",
        "National security policies", "Trade agreements and foreign policy", "Political scandals and accountability",
        "Global governance and institutions", "Impact of lobbying", "Decentralized governance",
        "Electoral reforms", "Civic engagement initiatives", "Populism and its effects"
    ],
    'Education': [
        "Literacy rates", "Online education trends", "Student loan",
        "STEM education initiatives", "Role of AI in personalized learning", "Education inequality and access",
        "Teacher training and recruitment", "Importance of arts and humanities", "International student mobility",
        "Higher education funding", "School curriculum reforms", "Special education advancements",
        "Global education rankings", "Parent involvement in education", "Technology in classrooms"
    ]
    
    }
    
    scraper = WikipediaScraper()
    scraper.process_topics(topics)