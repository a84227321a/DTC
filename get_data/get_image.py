from icrawler.builtin import BaiduImageCrawler
from icrawler.builtin import BingImageCrawler
from icrawler.builtin import GoogleImageCrawler

# # 谷歌
# google_storage = {'root_dir':r'H:\data\yinzhang\google'}
# google_crawler = GoogleImageCrawler(parser_threads=4,downloader_threads=4,storage=google_storage)
# google_crawler.crawl(keyword='公章',
#                      max_num=10)


#必应
bing_storage = {'root_dir': r'H:\data\yinzhang\biying_1'}
bing_crawler = BingImageCrawler(parser_threads=2,
                                downloader_threads=4,
                                storage=bing_storage)
bing_crawler.crawl(keyword='公章',
                   max_num=5000)

bing_storage_2 = {'root_dir': r'H:\data\yinzhang\biying_2'}
bing_crawler = BingImageCrawler(parser_threads=2,
                                downloader_threads=4,
                                storage=bing_storage_2)
bing_crawler.crawl(keyword='公司章',
                   max_num=5000)


bing_storage_2 = {'root_dir': r'H:\data\yinzhang\biying_3'}
bing_crawler = BingImageCrawler(parser_threads=2,
                                downloader_threads=4,
                                storage=bing_storage_2)
bing_crawler.crawl(keyword='合同章',
                   max_num=5000)
# #百度
# baidu_storage = {'root_dir': r'H:\data\yinzhang\baidu'}
#
# baidu_crawler = BaiduImageCrawler(parser_threads=2,
#                                   downloader_threads=4,
#                                   storage=baidu_storage)
# baidu_crawler.crawl(keyword='公章',
#                     max_num=1000)