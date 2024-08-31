class Solution{
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        // m for length of nums1, n for length of nums2
        //num1 have spaces m+n
        if (m==0){
            nums1=nums2;
            return;
        }
        if (n==0){
            return;
        }
        int idx1=m-1;
        int idx2=n-1;
        int i;
        for (i=m+n-1; i>=0; i--){
            if (nums1[idx1]>nums2[idx2]){
                nums1[i]=nums1[idx1];
                if (idx1==0){
                break;
                }else{
                    idx1--;
                }            
            }
            else{
                nums1[i]=nums2[idx2];
                if (idx2==0){
                    idx2=-1;
                    break;
                }else{
                    idx2--;
                }  
            }
        }            
        while (idx2>=0){
            nums1[i]=nums2[idx2];
            idx2--;
            i--;
        }
    }
};
