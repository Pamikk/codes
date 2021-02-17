/*A password is considered strong if the below conditions are all met:

It has at least 6 characters and at most 20 characters.
It contains at least one lowercase letter, at least one uppercase letter, and at least one digit.
It does not contain three repeating characters in a row (i.e., "...aaa..." is weak, but "...aa...a..." is strong, assuming other conditions are met).
Given a string password, return the minimum number of steps required to make password strong. if password is already strong, return 0.

In one step, you can:

Insert one character to password,
Delete one character from password, or
Replace one character of password with another character.*/
class Solution {
public:
    int strongPasswordChecker(string password) {
        int l = password.length();
        unsigned short lower = 0;
        unsigned short upper = 0;
        unsigned short digit = 0;
        int reps[16];
        rep = 0;
        int replaces=0;
        int repeat=0;
        while (i<l){
            int tmp = (int)password[i];
            if (65<=tmp<=90){
                upper=1;
            }else if(97<=tmp<=122){
                lower=1;
            }else if (48<=tmp<=57){
                digit=1;
            }
            j=i+1;
            while((j<l)&&(password[j]==password[i])){
                j++;
            }
            if ((j-i)>2){
                k=rep;
                while(k>0){
                    if (reps[k-1]<j-i){
                        reps[k] = reps[k-1];
                        k--;
                    }else{
                        break;
                    }
                }
                reps[k]=j-i;
                rep+=1;
                replaces += (j-i)/3;
            }
            i=j;
        }
        unsigned short lv = upper+lower+digit;
        if (lv==3){
            if (rep==0){
                if (6<=l<=20){
                    return 0;
                }else{
                    return max(l-20,6-l);
                }
            }else{
                if (6<=l<=20){
                    return replaces;
                }else if (l<6){
                    return 6-l;
                }else{

                }
            }
        }

    }
};