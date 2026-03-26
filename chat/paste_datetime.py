#! /usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 11:49:18 2021

@author: JOHN WARUTUMO
"""
import time
import pyperclip

if __name__ == '__main__':
    ts = time.strftime('%Y%m%d%H%M', time.gmtime(time.time()))
    pyperclip.copy('{'+ts+':\n\n}')
    pyperclip.paste()

'''
0	1	1	1	1
1	1	1	1	1
2	4	3	3	2
3	29	19	9	5
4	355	219	33	16
5	6942	4231	139	63
6	209527	130023	718	318
7	9535241	6129859	4535	2045
8	642779354	431723379	35979	16999
9	63260289423	44511042511	363083	183231
10	 	6611065248783	4717687	2567284
'''

'''
1. open-mind
2. automation
3. planning capacity
4. curiosity
5. clarity of purpose
'''
