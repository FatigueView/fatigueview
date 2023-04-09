# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2020/9/15 2:38 PM
=================================================='''
import numpy
from scipy.optimize import fsolve





class ldmk_transform:
    def __init__(self,base_ldmk,ldmk_num = 28):
        self.ldmk_num = ldmk_num
        self.base_ldmk = base_ldmk

        self.origin_corner = self._get_corner(self.base_ldmk)


    def _get_corner(self,ldmk):
        ldmk_id =[6,13,21,19]
        if self.ldmk_num == 68:
            ldmk_id = [37,46,55,49]

        corner_ldmk = [ldmk[i] for i in ldmk_id]
        return corner_ldmk


    def _solve_h(self,h):
        h = list(map(float,h))
        ans = []
        for ldmk,origin_ldmk in zip(self.corner,self.origin_corner):
            temp_ans = h[0] * origin_ldmk[0]+ h[1] * origin_ldmk[1]+ h[2]
            temp_ans /= (h[7] * origin_ldmk[0]+ h[8] * origin_ldmk[1]+ 1)
            ans.append(temp_ans)
            temp_ans = h[0] * origin_ldmk[0] + h[1] * origin_ldmk[1] + h[2]
            temp_ans /= (h[7] * origin_ldmk[0] + h[8] * origin_ldmk[1] + 1)
            ans.append(temp_ans)
        return ans



    def _get_transform(self,corner,base_corner,h):
        A = []

    def transfom_ldmk(self,ldmk):
        corner = self._get_corner(ldmk)
        self.corner = corner

        init_h = [1,0,0,0,1,0,0,0]
        h = fsolve(self._solve_h,init_h)






