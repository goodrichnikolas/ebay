import asyncio
from playwright.async_api import async_playwright, Page, Browser, Playwright
import pandas as pd
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import re
from urllib.parse import quote_plus
from dataclasses import dataclass, asdict
import aiofiles
import aiohttp
import sys
from contextlib import asynccontextmanager

@dataclass
class EbayItem:
    """Data class to store eBay item information."""
    title: str
    price: float
    shipping: Optional[str]
    sold_date: Optional[str]
    img_link: Optional[str]
    item_page: str
    condition: Optional[str]
    scrape_date: str

class EbayPlaywrightScraper:
    def __init__(self, search_term: str, output_dir: str = "output", headless: bool = True):
        self.search_term = search_term
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.headless = headless
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.setup_directories()
        self.setup_logging()

    def setup_directories(self) -> None:
        self.output_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)

    def setup_logging(self) -> None:
        """Configure logging with both file and console output."""
        # Clear any existing handlers
        logging.getLogger().handlers = []
        
        # Create formatters and handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(
            self.output_dir / f'scraper_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Set up the root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    @staticmethod
    def make_safe_filename(string: str) -> str:
        safe_string = re.sub(r'[<>:"/\\|?*]', '_', string)
        return safe_string[:100]

    @asynccontextmanager
    async def get_browser(self):
        """Context manager for browser initialization and cleanup."""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=['--disable-dev-shm-usage']
            )
            yield self.browser
        finally:
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()

    async def create_page(self) -> Page:
        if not self.browser:
            raise RuntimeError("Browser not initialized")
            
        page = await self.browser.new_page()
        await page.set_viewport_size({"width": 1920, "height": 1080})
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
        return page

    async def get_page_items(self, page: Page) -> List[EbayItem]:
        try:
            # Wait for the items to load
            await page.wait_for_selector('.s-item__wrapper', timeout=30000)
            
            # Extract items using evaluate
            items_data = await page.evaluate("""
                () => {
                    const items = document.querySelectorAll('.s-item__wrapper');
                    return Array.from(items).map(item => {
                        const titleElem = item.querySelector('.s-item__title');
                        const priceElem = item.querySelector('.s-item__price');
                        const shippingElem = item.querySelector('.s-item__shipping');
                        const soldDateElem = item.querySelector('.POSITIVE');
                        const imgElem = item.querySelector('.s-item__image-img');
                        const linkElem = item.querySelector('.s-item__link');
                        const conditionElem = item.querySelector('.SECONDARY_INFO');
                        
                        return {
                            title: titleElem ? titleElem.textContent : null,
                            price: priceElem ? priceElem.textContent : null,
                            shipping: shippingElem ? shippingElem.textContent : null,
                            soldDate: soldDateElem ? soldDateElem.textContent : null,
                            imgLink: imgElem ? imgElem.src : null,
                            itemPage: linkElem ? linkElem.href : null,
                            condition: conditionElem ? conditionElem.textContent : null
                        };
                    });
                }
            """)

            items = []
            for item_data in items_data:
                if not item_data['title'] or item_data['title'] == 'Shop on eBay':
                    continue

                price_str = item_data['price']
                price = float(re.sub(r'[^\d.]', '', price_str)) if price_str else 0.0

                item = EbayItem(
                    title=self.make_safe_filename(item_data['title']),
                    price=price,
                    shipping=item_data['shipping'],
                    sold_date=item_data['soldDate'],
                    img_link=item_data['imgLink'],
                    item_page=item_data['itemPage'],
                    condition=item_data['condition'],
                    scrape_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                items.append(item)

            return items
        except Exception as e:
            logging.error(f"Error extracting items from page: {str(e)}")
            return []

    async def download_images(self, df: pd.DataFrame) -> None:
        async with aiohttp.ClientSession() as session:
            tasks = []
            for _, row in df.iterrows():
                if row['img_link'] and row['title']:
                    img_path = self.images_dir / f"{row['title']}.jpg"
                    if not img_path.exists():  # Skip if image already exists
                        tasks.append(self.download_image(session, row['img_link'], row['title']))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                successful = sum(1 for r in results if isinstance(r, bool) and r)
                logging.info(f"Downloaded {successful} out of {len(tasks)} images")

    async def download_image(self, session: aiohttp.ClientSession, img_url: str, filename: str) -> bool:
        try:
            async with session.get(img_url) as response:
                if response.status == 200:
                    img_path = self.images_dir / f"{filename}.jpg"
                    async with aiofiles.open(img_path, 'wb') as f:
                        await f.write(await response.read())
                    return True
            return False
        except Exception as e:
            logging.error(f"Failed to download image {img_url}: {str(e)}")
            return False

    async def scrape(self) -> pd.DataFrame:
        """Main scraping function."""
        all_items = []
        
        async with self.get_browser() as browser:
            try:
                page = await self.create_page()
                page_num = 1
                
                while True:
                    logging.info(f"Scraping page {page_num}")
                    
                    encoded_term = quote_plus(self.search_term)
                    url = f'https://www.ebay.com/sch/i.html?_from=R40&_nkw={encoded_term}&_sacat=0&rt=nc&LH_Sold=1&LH_Complete=1'
                    if page_num > 1:
                        url += f'&_pgn={page_num}'

                    try:
                        await page.goto(url, wait_until='networkidle')
                        items = await self.get_page_items(page)
                        
                        if not items:
                            logging.info("No items found on page, ending scrape")
                            break
                            
                        all_items.extend(items)
                        logging.info(f"Found {len(items)} items on page {page_num}")
                        
                        next_button = await page.query_selector('.pagination__next')
                        if not next_button:
                            logging.info("No next page button found, ending scrape")
                            break
                            
                        page_num += 1
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logging.error(f"Error processing page {page_num}: {str(e)}")
                        break

                df = pd.DataFrame([asdict(item) for item in all_items])
                
                if not df.empty:
                    csv_path = self.output_dir / f'{self.search_term}_{datetime.now().strftime("%Y%m%d")}.csv'
                    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                    logging.info(f"Results saved to {csv_path}")
                    
                    await self.download_images(df)
                
                return df
                
            except Exception as e:
                logging.error(f"Scraping failed: {str(e)}")
                raise

async def main():
    try:
        search_term = input("Enter search term: ")
        scraper = EbayPlaywrightScraper(search_term, headless=False)
        
        df = await scraper.scrape()
        print(f"\nScraped {len(df)} items")
        if not df.empty:
            print("\nSample of results:")
            print(df.head())
            
    except Exception as e:
        logging.error(f"Script failed: {str(e)}", exc_info=True)
        print(f"An error occurred. Check the log file in the output directory for details.")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())