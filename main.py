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
        """Create output and image directories if they don't exist."""
        self.output_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)

    def setup_logging(self) -> None:
        """Configure logging to both file and console."""
        logging.getLogger().handlers = []
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler(
            self.output_dir / f'scraper_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    @staticmethod
    def make_safe_filename(string: str) -> str:
        """Convert string to safe filename by removing invalid characters."""
        safe_string = re.sub(r'[<>:"/\\|?*]', '_', string)
        return safe_string[:100]

    async def scroll_to_bottom(self, page: Page) -> None:
        """Scroll to the bottom of the page to ensure all content is loaded."""
        try:
            # Get initial page height
            last_height = await page.evaluate('document.body.scrollHeight')
            
            while True:
                # Scroll to bottom
                await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                # Wait for potential dynamic content to load
                await asyncio.sleep(2)
                
                # Calculate new scroll height
                new_height = await page.evaluate('document.body.scrollHeight')
                
                # Break if no more content is loading
                if new_height == last_height:
                    break
                    
                last_height = new_height
                
            # Scroll back to top to ensure consistent state
            await page.evaluate('window.scrollTo(0, 0)')
            await asyncio.sleep(1)
            
        except Exception as e:
            logging.error(f"Error during page scrolling: {str(e)}")

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
        """Create and configure a new browser page."""
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

    async def check_for_captcha(self, page: Page) -> bool:
        """Check if the page contains a CAPTCHA and handle it."""
        captcha_selector = 'input[name="captcha"]'
        try:
            captcha = await page.wait_for_selector(captcha_selector, timeout=2000)
            if captcha:
                logging.warning("CAPTCHA detected! Waiting for manual solve...")
                await page.wait_for_selector(captcha_selector, state="hidden", timeout=300000)  # 5 minute timeout
                return True
        except:
            return False
        return False

    async def get_page_items(self, page: Page) -> List[EbayItem]:
        """Extract items from the current page."""
        try:
            await page.wait_for_selector('.s-item__wrapper', timeout=30000)
            
            # Scroll to ensure all content is loaded
            await self.scroll_to_bottom(page)
            
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

                try:
                    price_str = item_data['price']
                    if price_str:
                        # Remove currency symbols and spaces
                        price_str = re.sub(r'[^\d.,]', '', price_str)
                        
                        # Handle different decimal/thousand separators
                        # If there are multiple dots/commas, assume the last one is the decimal separator
                        parts = re.split(r'[.,]', price_str)
                        
                        if len(parts) > 1:
                            # Join all parts except the last one (removing all separators)
                            # and add the last part as decimal
                            whole_part = ''.join(parts[:-1])
                            decimal_part = parts[-1]
                            price_str = f"{whole_part}.{decimal_part}"
                        
                        # Convert to float
                        price = float(price_str)
                    else:
                        price = 0.0
                        
                except Exception as e:
                    logging.warning(f"Error parsing price '{item_data['price']}': {str(e)}. Setting price to 0.0")
                    price = 0.0

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
        """Download images for all items in the DataFrame."""
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
        """Download a single image."""
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
        """Main scraping function with improved pagination."""
        all_items = []
        retry_count = 0
        max_retries = 3
        
        async with self.get_browser() as browser:
            try:
                page = await self.create_page()
                page_num = 1
                
                encoded_term = quote_plus(self.search_term)
                base_url = f'https://www.ebay.com/sch/i.html?_from=R40&_nkw={encoded_term}&_sacat=0&rt=nc&LH_Sold=1&LH_Complete=1'
                
                while True:
                    logging.info(f"Scraping page {page_num}")
                    
                    url = f"{base_url}&_pgn={page_num}" if page_num > 1 else base_url
                    
                    try:
                        await page.goto(url)
                        await asyncio.sleep(2)
                        
                        if await self.check_for_captcha(page):
                            continue
                        
                        # Get items after ensuring page is fully loaded
                        items = await self.get_page_items(page)
                        
                        # Check for next button using the correct selector
                        next_button = await page.query_selector('.pagination__next:not([aria-disabled="true"])')
                        
                        if not items:
                            if retry_count < max_retries:
                                retry_count += 1
                                logging.warning(f"No items found, retrying ({retry_count}/{max_retries})...")
                                await asyncio.sleep(5 * retry_count)
                                continue
                            else:
                                logging.info("No more items found")
                                break
                        
                        retry_count = 0  # Reset retry count on successful page
                        all_items.extend(items)
                        logging.info(f"Found {len(items)} items on page {page_num}")
                        
                        if page_num % 5 == 0:
                            self.save_intermediate_results(all_items)
                        
                        if not next_button:
                            logging.info("No next button found - reached last page")
                            break
                            
                        page_num += 1
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        logging.error(f"Error processing page {page_num}: {str(e)}")
                        if retry_count < max_retries:
                            retry_count += 1
                            await asyncio.sleep(5 * retry_count)
                            continue
                        else:
                            break

                df = pd.DataFrame([asdict(item) for item in all_items])
                
                if not df.empty:
                    self.save_final_results(df)
                    await self.download_images(df)
                
                return df
                
            except Exception as e:
                logging.error(f"Scraping failed: {str(e)}")
                raise

    def save_intermediate_results(self, items: List[EbayItem]) -> None:
        """Save intermediate results to CSV."""
        df = pd.DataFrame([asdict(item) for item in items])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        interim_path = self.output_dir / f'{self.search_term}_interim_{timestamp}.csv'
        df.to_csv(interim_path, index=False, encoding='utf-8-sig')
        logging.info(f"Saved interim results to {interim_path}")

    def save_final_results(self, df: pd.DataFrame) -> None:
        """Save final results to CSV."""
        csv_path = self.output_dir / f'{self.search_term}_final_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logging.info(f"Final results saved to {csv_path}")

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