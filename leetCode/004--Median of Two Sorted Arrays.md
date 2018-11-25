
```
There are two sorted arrays nums1 and nums2 of size m and n respectively.

Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).

You may assume nums1 and nums2 cannot be both empty.

Example 1:

nums1 = [1, 3]
nums2 = [2]

The median is 2.0
Example 2:

nums1 = [1, 2]
nums2 = [3, 4]

The median is (2 + 3)/2 = 2.5
```


```python
# 在没有时间复杂度限制的情况下可以随便撸
class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        nums1.extend(nums2)
        nums1.sort()
        if len(nums1)%2 == 0:
            half0 = len(nums1) // 2  - 1 
            half1 = len(nums1) // 2  
            return (nums1[half0] + nums1[half1]) / 2.
        if len(nums1)%2 == 1:
            half = len(nums1) // 2  
            return nums1[half]
```


```python
# 但是是有时间复杂度限制的。 O(log(m+n))


```
