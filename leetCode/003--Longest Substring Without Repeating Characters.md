
```
Given a string, find the length of the longest substring without repeating characters.

Example 1:

Input: "abcabcbb"
Output: 3 
Explanation: The answer is "abc", with the length of 3. 
Example 2:

Input: "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
Example 3:

Input: "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3. 
             Note that the answer must be a substring, "pwke" is a subsequence and not a substring.
```

```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """

        d = {}
        longest=0
        cur = 0
        cur_length = 0
        while True:
            if cur == len(s):
                return max(cur_length, longest)
            elif s[cur] not in d:
                d[s[cur]] = cur
                cur = cur + 1
                cur_length = cur_length + 1
            else:
                if cur_length > longest:
                    longest = cur_length
                cur = d[s[cur]] + 1
                d = {}
                cur_length = 0

```
