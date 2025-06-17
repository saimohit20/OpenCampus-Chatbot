import asyncio
from playwright.async_api import async_playwright
import json

async def get_course_details_single_test_full(page, course_url):
    """
    Navigates to a specific course URL and extracts detailed information
    including all the sections identified:
    Description, Mentor (now handles multiple), What We Will Learn, Day/Time Summary, ECTS,
    Location, Language, Combined Information (What's In It For You + Prior Knowledge), and Sessions.

    Args:
        page (playwright.async_api.Page): The Playwright page object (reused).
        course_url (str): The URL of the individual course page.

    Returns:
        dict: A dictionary containing the extracted course details.
    """
    course_details = {
        'description': None,
        'mentor': [],  # Changed to an empty list to store multiple mentors
        'what_we_will_learn': [],
        'day_time_summary': None,
        'ects': None,
        'location': None,
        'language': None,
        'information': None,
        'course_sessions': [],
    }
    print(f"\n--- Attempting to fetch ALL details from: {course_url} ---")

    try:
        await page.goto(course_url, wait_until="networkidle", timeout=30000)
        await page.wait_for_load_state('domcontentloaded')
        print("   Page loaded successfully.")

        # --- 1. Course Description ---
        description_locator = page.locator(".flex.flex-1.flex-col.text-white.mb-4.lg\\:mb-20 span.text-2xl.mt-2").first
        if await description_locator.is_visible():
            course_details['description'] = (await description_locator.text_content()).strip()
            print(f"   Description: {course_details['description'][:70]}...")
        else:
            print("   Description not found or not visible.")
            course_details['description'] = None

        # --- 2. Course Mentor (Updated to handle multiple mentors) ---
        mentor_name_locators = await page.locator("div.flex.flex-col.text-lg span.mb-1").all() # Use .all()
        if mentor_name_locators:
            course_details['mentor'] = [(await locator.text_content()).strip() for locator in mentor_name_locators]
            print(f"   Mentors: {', '.join(course_details['mentor'])}")
        else:
            print("   Mentor(s) not found or not visible.")
            course_details['mentor'] = [] # Ensure it's an empty list if no mentors found

        # --- 3. What we will learn ---
        learn_elements = await page.locator("ul.list-disc.pb-12 li div.ml-2 span").all()
        if learn_elements:
            course_details['what_we_will_learn'] = [(await el.text_content()).strip() for el in learn_elements if await el.text_content()]
            print(f"   What we will learn ({len(course_details['what_we_will_learn'])} points):")
            for i, item in enumerate(course_details['what_we_will_learn']):
                print(f"     - {item}")
        else:
            print("   'What we will learn' section not found or no items visible.")
            course_details['what_we_will_learn'] = []

        # --- 4. Course Day & Time (Summary) ---
        day_time_summary_locator = page.locator(".flex.flex-1.flex-col.text-white.mb-4.lg\\:mb-20 > span.text-xs").first
        if await day_time_summary_locator.is_visible():
            course_details['day_time_summary'] = (await day_time_summary_locator.text_content()).strip()
            print(f"   Day & Time Summary: {course_details['day_time_summary']}")
        else:
            print("   Day & Time summary not found or not visible.")
            course_details['day_time_summary'] = None

        # --- 5. Course ECTS ---
        ects_label_locator = page.locator('span.text-lg.mt-2.text-center', has_text='ECTS').first
        if await ects_label_locator.is_visible():
            ects_value = await ects_label_locator.evaluate('''node => {
                // The structure is: [ECTS label] [19:00 - 20:30 span] [2,5 span]
                let current = node;
                if (current && current.nextElementSibling) {
                    current = current.nextElementSibling;
                    if (current && current.nextElementSibling) {
                        return current.nextElementSibling.textContent.trim();
                    }
                }
                return null;
            }''')
            course_details['ects'] = ects_value if ects_value else None
            print(f"   ECTS: {course_details['ects']}")
        else:
            print("   ECTS label not found or not visible.")
            course_details['ects'] = None

        # --- 6. Location & Language taught ---
        location_locator = page.locator('span.text-sm.mt-2.text-center', has_text='KIEL').first
        if await location_locator.is_visible():
            course_details['location'] = (await location_locator.text_content()).strip()
            print(f"   Location: {course_details['location']}")
        else:
            # Try to find 'ONLINE' if 'KIEL' is not found
            online_location_locator = page.locator('span.text-sm.mt-2.text-center', has_text='ONLINE').first
            if await online_location_locator.is_visible():
                course_details['location'] = (await online_location_locator.text_content()).strip()
                print(f"   Location: {'Found (Online)' if course_details['location'] else 'Empty'}")
            else:
                print("   Location: Neither 'KIEL' nor 'ONLINE' found.")
            course_details['location'] = None

        language_locator = page.locator('span.text-sm.mt-2.text-center', has_text='ENGLISCH').first
        if await language_locator.is_visible():
            course_details['language'] = (await language_locator.text_content()).strip()
            print(f"   Language: {course_details['language']}")
        else:
            # Try to find 'DEUTSCH' if 'ENGLISCH' is not found
            german_language_locator = page.locator('span.text-sm.mt-2.text-center', has_text='DEUTSCH').first
            if await german_language_locator.is_visible():
                course_details['language'] = (await german_language_locator.text_content()).strip()
                print(f"   Language: {'Found (Deutsch)' if course_details['language'] else 'Empty'}")
            else:
                print("   Language: Neither 'ENGLISCH' nor 'DEUTSCH' found.")
            course_details['language'] = None
        
        # --- Combined Information Section (What's in It for You? & Prior Knowledge) ---
        # Target the main wrapping div by its specific classes
        combined_info_locator = page.locator(".flex.flex-wrap.bg-edu-course-invited.rounded-2xl.p-4.mx-6.xl\\:mx-0").first
        
        if await combined_info_locator.is_visible():
            full_text = await combined_info_locator.text_content()
            course_details['information'] = ' '.join(full_text.split()).strip()
            print(f"   Combined Information Section: {course_details['information'][:150]}...")
        else:
            print("   Combined Information section container not found.")
            course_details['information'] = None
        
        # --- 9. Course Sessions ---
        course_sessions_data = []
        session_list_items = await page.locator('ul.max-w-2xl li.flex.mb-4').all()
        
        if session_list_items:
            print(f"   Found {len(session_list_items)} course sessions.")
            for i, item in enumerate(session_list_items):
                session = {}
                # Date
                date_span = item.locator('span.block.text-sm.sm\\:text-lg.font-semibold').first
                session['date'] = (await date_span.text_content()).strip() if await date_span.is_visible() else None

                # Time
                time_span = item.locator('span.text-sm.whitespace-nowrap').first
                session['time'] = (await time_span.text_content()).strip() if await time_span.is_visible() else None

                # Topic
                topic_span = item.locator('span.block.text-sm.sm\\:text-lg.break-words').first
                session['topic'] = (await topic_span.text_content()).strip() if await topic_span.is_visible() else None

                # Location Text & URL
                location_link = item.locator('span.text-sm.text-gray-400.ml-0.pl-0 a').first
                if await location_link.is_visible():
                    session['location_text'] = (await location_link.text_content()).strip()
                    session['location_url'] = await location_link.get_attribute('href')
                else:
                    location_span_text = item.locator('span.text-sm.text-gray-400.ml-0.pl-0').first
                    session['location_text'] = (await location_span_text.text_content()).strip() if await location_span_text.is_visible() else None
                    session['location_url'] = None
                
                course_sessions_data.append(session)
                print(f"     Session {i+1}: Date={session['date']}, Topic={session['topic']}")
            course_details['course_sessions'] = course_sessions_data
        else:
            print("   Course sessions not found.")
            course_details['course_sessions'] = []

    except Exception as e:
        print(f"   An error occurred during detail extraction for {course_url}: {e}")
    
    return course_details

async def run_single_course_test_full():
    test_url = "https://edu.opencampus.sh/en/course/559" # You can change this URL to test others, e.g., one with multiple mentors
    print(f"Starting comprehensive test for: {test_url}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True) # Set to False if you want to see the browser window
        page = await browser.new_page()

        extracted_data = await get_course_details_single_test_full(page, test_url)
        
        await browser.close()
        print("\n--- Comprehensive Single Course Test Complete ---")
        print("\nExtracted Data (JSON format):")
        print(json.dumps(extracted_data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(run_single_course_test_full())