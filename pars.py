import json
import httpx
from urllib import parse
from bs4 import BeautifulSoup


class MEM_PARS:
    def __init__(self):
        self.client = httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)
        )

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

    def get_bio(self, links: list) -> dict:
        url = "https://memexpert.net"
        result_dict = []

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

            result_dict.append(_dict)

        return json.dumps(result_dict, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = MEM_PARS()

    print(parser.get_bio(parser.get_links("писос")))
