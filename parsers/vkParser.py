from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import time
import os
import re
import json
import requests
from typing import List, Dict, Any, Optional


class VK:
    """Парсер VK для сбора постов, изображений и информации о постах."""

    def __init__(self, config: Dict[str, Any]):
        """Инициализация парсера и настройка webdriver.

        :param config: Конфигурация (см. док-строку класса)
        """
        self.urls: List[str] = config.get("urls", [])
        if not isinstance(self.urls, list) or not self.urls:
            raise ValueError("config must include 'urls' as a non-empty list")

        self.min_image_size_kb: float = float(config.get("min_image_size_kb", 2))
        self.scroll_pause_time: float = float(config.get("scroll_pause_time", 7))
        self.max_scrolls: int = int(config.get("max_scrolls", 3))
        self.chrome_args: List[str] = (
            config.get(
                "chrome_args",
                [
                    "--window-size=1400,1000",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--headless",
                ],
            )
            or []
        )
        self.output_dir: str = config.get("output_dir", ".")

        os.makedirs(self.output_dir, exist_ok=True)

        options = Options()
        for a in self.chrome_args:
            options.add_argument(a)
        options.add_experimental_option("detach", True)

        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )

    # utils
    def is_valid_text(self, txt: Optional[str]) -> bool:
        """Проверяет, не сломана ли кодировка текста.

        Возвращает True, если более 70% символов относятся к ожидаемому набору
        (русские/латинские буквы, цифры, пробелы и базовая пунктуация).
        """
        if not txt or len(txt.strip()) < 5:
            return 0
        cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", txt)
        read = re.findall(r'[A-Za-zА-Яа-яЁё0-9\s.,!?;:«»"\'()\-–…]', cleaned)
        return (len(read) / max(len(cleaned), 1)) > 0.7

    def try_fix_encoding(self, s: Optional[str]) -> Optional[str]:
        """Восстановление кодировки строки (latin1/cp1251 -> utf-8)."""
        if not s:
            return s
        cods = [
            ("latin1", "utf-8"),
            ("cp1251", "utf-8"),
        ]
        for frm, to in cods:
            try:
                fixed = s.encode(frm).decode(to)
                if self.is_valid_text(fixed):
                    return fixed
            except Exception:
                pass
        return s

    def get_image_size_kb(self, url: str) -> float:
        """Возвращает размер изображения в КБ."""
        try:
            r = requests.head(url, timeout=10, allow_redirects=True)
            size = r.headers.get("Content-Length")
            if size:
                return int(size) / 1024
        except Exception:
            pass

        try:
            r = requests.get(url, stream=True, timeout=15)
            r.raise_for_status()
            total = 0
            for chunk in r.iter_content(1024 * 8):
                if not chunk:
                    break
                total += len(chunk)
            return total / 1024
        except Exception:
            return 0.0

    def download_image(self, url: str, outpath: str) -> bool:
        """Скачивает изображение по URL в указанный файл. Возвращает True при успехе."""
        try:
            r = requests.get(url, stream=True, timeout=15)
            r.raise_for_status()
            with open(outpath, "wb") as f:
                for chunk in r.iter_content(1024):
                    if not chunk:
                        break
                    f.write(chunk)
            return 1
        except Exception:
            return 0

    def safe_name(self, s: str) -> str:
        """Формирует имя для папки/файла из строки."""
        s = re.sub(r"https?://", "", s)
        s = re.sub(r"[^A-Za-z0-9_-]", "_", s)
        return s.strip("_")[:120] or "community"

    def get_image_dimensions(self, image_url: str) -> Optional[Dict[str, int]]:
        """Возвращает размеры изображения (ширина и высота) по URL."""
        try:
            r = requests.get(image_url, stream=True, timeout=15)
            r.raise_for_status()
            image = Image.open(BytesIO(r.content))
            width, height = image.size
            return {"width": width, "height": height}
        except Exception:
            return None

    # logic
    def collect_posts_from_page(self, pub_name: str) -> List[Dict[str, Any]]:
        """Собирает посты со страницы, которая уже загружена в self.driver."""
        post_data: List[Dict[str, Any]] = []

        try:
            wait = WebDriverWait(self.driver, 20)
            wait.until(EC.presence_of_element_located((By.ID, "page_wall_posts")))
        except Exception:
            pass

        time.sleep(1)

        post_elems = self.driver.find_elements(
            By.CSS_SELECTOR,
            "#page_wall_posts div[data-post-id], #page_wall_posts div[id^='post-'], div[id^='post-']",
        )

        for el in post_elems:
            try:
                pid = el.get_attribute("data-post-id") or el.get_attribute("id") or ""

                txt = ""
                selectors = [
                    ".vkitShowMoreText__text--QzxEF",
                    ".vkitTextClamp__root--8Ttiw",
                    ".vkitPostText__root--otCAj",
                    ".wall_post_text",
                    '[data-testid^="showmoretext"]',
                ]
                for sel in selectors:
                    sub = el.find_elements(By.CSS_SELECTOR, sel)
                    if sub:
                        txt = (
                            sub[0].get_attribute("innerText")
                            or sub[0].get_attribute("textContent")
                            or ""
                        )
                        if txt and txt.strip():
                            break
                if not txt or not txt.strip():
                    txt = (
                        el.get_attribute("innerText")
                        or el.get_attribute("textContent")
                        or ""
                    )

                if not self.is_valid_text(txt):
                    fixed = self.try_fix_encoding(txt)
                    if self.is_valid_text(fixed):
                        txt = fixed
                    else:
                        continue

                imgs: List[str] = []
                img_els = el.find_elements(By.CSS_SELECTOR, "img")
                for img_el in img_els:
                    cls = img_el.get_attribute("class") or ""
                    if "AvatarRich" in cls or "post_field_user_image" in cls:
                        continue
                    src = (
                        img_el.get_attribute("src")
                        or img_el.get_attribute("data-src")
                        or img_el.get_attribute("data-lazy-src")
                    )
                    if not src:
                        continue
                    if src.startswith("//"):
                        src = "https:" + src
                    if src.startswith("http") and src not in imgs:
                        imgs.append(src)

                if not imgs:
                    continue

                main_img = max(imgs, key=lambda u: len(u))

                size_kb = self.get_image_size_kb(main_img)
                if size_kb < self.min_image_size_kb:
                    continue

                dimensions = self.get_image_dimensions(main_img)

                ext = os.path.splitext(main_img.split("?")[0])[-1] or ".jpg"
                if not re.match(r"\.(jpg|jpeg|png|webp|gif)$", ext, re.I):
                    ext = ".jpg"

                safe_pid = self.safe_name(pid)
                filename = f"{safe_pid}_1{ext}"

                images_dir = os.path.join(self.output_dir, f"images_vk")
                os.makedirs(images_dir, exist_ok=True)
                path = os.path.join(images_dir, filename)

                downloaded = []
                if self.download_image(main_img, path):
                    downloaded.append(path)
                else:
                    downloaded.append(main_img)

                post_data.append(
                    {
                        "post_id": pid,
                        "text": txt.strip(),
                        "image_name": filename,
                        "image_link": main_img,
                        "description": txt.strip(),
                        "width": dimensions.get("width") if dimensions else None,
                        "height": dimensions.get("height") if dimensions else None,
                        "image_size_kb": round(size_kb, 1),
                        "downloaded_images": downloaded,
                    }
                )

            except Exception:
                continue

        return post_data

    def scroll_page(self):
        """Прокручивает текущую страницу вниз, пока не достигнет конца или не превысит max_scrolls."""
        last_h = self.driver.execute_script("return document.body.scrollHeight")
        for i in range(self.max_scrolls):
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            time.sleep(self.scroll_pause_time)
            new_h = self.driver.execute_script("return document.body.scrollHeight")
            if new_h == last_h:
                print(f"Конец ленты на {i+1}")
                break
            last_h = new_h
            print(f"Прокрутили {i+1}/{self.max_scrolls}")

    def process_community(self, url: str):
        """Открывает сообщество по URL, прокручивает страницу, собирает посты и сохраняет JSON."""
        print(f"=== {url} ===")
        community_raw = url.rstrip("/").split("/")[-1] or url
        pub_name = self.safe_name(community_raw)

        images_dir = os.path.join(self.output_dir, f"images_vk")
        json_dir = os.path.join(self.output_dir, f"jsons_vk")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)

        try:
            self.driver.get(url)
            try:
                wait = WebDriverWait(self.driver, 20)
                wait.until(EC.presence_of_element_located((By.ID, "page_wall_posts")))
            except Exception:
                pass

            time.sleep(2)
            try:
                webdriver.ActionChains(self.driver).send_keys(Keys.ESCAPE).perform()
            except Exception:
                pass

            self.scroll_page()

            post_data = self.collect_posts_from_page(pub_name)

            for post in post_data:
                post_json_path = os.path.join(json_dir, f"{post['post_id']}.json")
                with open(post_json_path, "w", encoding="utf-8") as f:
                    json.dump(post, f, ensure_ascii=False, indent=2)

            print(
                f"Сохранено {len(post_data)} постов в {json_dir} и картинки в {images_dir}"
            )

        except Exception as e:
            print(f"Ошибка при обработке {url}: {e}")

    def run(self):
        """Запускает обработку всех URL из конфигурации."""
        try:
            for u in self.urls:
                try:
                    self.process_community(u)
                except Exception as e:
                    print(f"Ошибка для {u}: {e}")
                    continue
        finally:
            try:
                self.driver.quit()
            except Exception:
                pass

        print("Done")


if __name__ == "__main__":
    cfg = {
        "urls": [
            "https://vk.com/poiskmemow",
            "https://vk.com/textmeme",
            "https://vk.com/memedescriptions",
            "https://vk.com/badtextmeme",
        ],
        "max_scrolls": 3000,
    }
    parser = VK(cfg)
    parser.run()
