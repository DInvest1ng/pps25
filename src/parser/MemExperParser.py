import json
import httpx
import os
from dotenv import load_dotenv
from urllib import parse
from bs4 import BeautifulSoup


load_dotenv()


class MEMEXPERT:
    def __init__(self):
        self.client = httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)
        )
        os.makedirs("images", exist_ok=True)
        os.makedirs("json_data", exist_ok=True)

    def load_image(self, url, title):
        response = self.client.get(url)
        if response.status_code == 200:
            ext = url.split(".")[-1].split("?")[0]
            filename = f"{title}.{ext}".replace("/", "_").replace("\\", "_")
            path = os.path.join("images", filename)
            with open(path, "wb") as f:
                f.write(response.content)
            return path
        return None

    def save_json(self, data, title):
        filename = f"{title}.json".replace("/", "_").replace("\\", "_")
        path = os.path.join("json_data", filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return path

    def get_links(self, text: str) -> list:
        url = "https://memexpert.net/search?q="
        response = self.client.get(url + parse.quote(text, encoding="utf-8"))
        soup = BeautifulSoup(response.text, "html.parser")
        links = [
            link.get("href")
            for link in soup.select('a[href^="/ru/"]')
            if link.get("href")
        ]
        return links

    def save_bio(self, links: list) -> None:
        url = "https://memexpert.net"
        for link in links:
            full_url = url + link
            response = self.client.get(full_url)
            soup = BeautifulSoup(response.content, "html.parser")
            _dict = {}
            desired_og_properties = [
                "og:title",
                "og:description",
                "og:url",
                "og:locale",
                "og:type",
                "og:image",
                "og:image:type",
                "og:image:width",
                "og:image:height",
                "og:image:alt",
            ]
            for prop in desired_og_properties:
                tag = soup.find("meta", property=prop)
                if tag and tag.get("content"):
                    content = tag["content"].strip()
                    _dict[prop.replace("og:", "")] = content

            self.load_image(_dict["image"], _dict["title"])
            self.save_json(_dict, _dict["title"])

        return None


if __name__ == "__main__":
    mem = MEMEXPERT()

    with open("ru_50k.txt", "r") as f:
        for line in f.readlines():
            try:
                mem.save_bio(mem.get_links(line.split(" ")))
            except:
                pass
