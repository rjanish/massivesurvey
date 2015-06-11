!This code uses the Dierckx libraries to fit a Bspline to VP sky data
!Important, the spline fit will NOT take values at identical x's, 
!this code searchs for these and averages them
Subroutine bspline_fit(NData_orig,num_pix,x,y,e_first,coeffs,k_spline,nest,&
knots,num_knots,s_mult,min_knot_space)
use wave_root
!Subroutine bspline_fit(NData,num_pix,x,y,s_mult,num_knots,knots,coeffs,k_spline,nest)
IMPLICIT NONE
INTEGER::num_knots,buffer1,knot_space,NData_flag,summed_pix,coll_num,same_wave,&
pix_run,NData_orig
REAL(KIND=4)::dummy1,dummy2,xb,xe,s_val,ls_fit,res_rms,res_mean,&
tmp_w,tmp_mean,tmp_err,x(NData_orig),y(NData_orig),e_first(NData_orig),&
xin(NData_orig),yin(NData_orig),coeffs(nest),knots(nest),s_mult,min_knot_space,&
spec_tot,srt_wrk(NData_orig),ein(NData_orig)
INTEGER::NData,status1,i,j,k,k_spline,iopt,nest,lwrk,&
ier,num_pix,Num_fit,num_repeat,multi_flag,cut_win,cut_val,&
NData2,skip_rms,skip_col,err1,err2,isrt_wrk(NData_orig)
INTEGER::iwrk(nest),skip_knot
REAL(KIND=4),ALLOCATABLE,DIMENSION(:)::w,wrk,&
x_spl,y_spl,rms_hold,x_spl2,y_spl2,w2,x_spl3,y_spl3,w3,&
x_spl4,y_spl4
pix_run=aperture*10
buffer1=4
!call sort(num_knots,knots)
NData=NData_orig
cut_win=11
cut_val=1.0
k_spline=3
Num_fit=nint(real(NData)/real(num_pix))
!iopt=0
iopt=-1
!s_val=1.0
!write(*,*) "Upon entry", NData, num_pix,Num_out
NData_flag=0
!Weeding out flag values
Do i=1,NData
   IF (nint(y(i)).ne.-666.and.y(i).lt.10.**12) THEN
      xin(i-NData_flag)=x(i)
      yin(i-NData_flag)=y(i)
      ein(i-NData_flag)=e_first(i)
   ELSE
      NData_flag=NData_flag+1
   END IF
!if (abs(x(i)-3519.148).lt.1.0) then
!write(*,*) i,x(i), y(i)
!end if
End Do
NData=NData-NData_flag
!Do i=1,NData
!   xin(i)=x(i)
!   yin(i)=y(i)
!End Do
!call sort2(NData,xin,yin)
call sort3(NData,xin,yin,ein,srt_wrk,isrt_wrk)
!open(83,file="full_sky.txt")
!Do i=1,NData
!   write(83,*) xin(i),yin(i)
!End Do
!Close(83)

!ALLOCATE(rms_hold(num_pix))
ALLOCATE(rms_hold(pix_run))
ALLOCATE(w(Num_fit))
ALLOCATE(x_spl(Num_fit))
ALLOCATE(y_spl(Num_fit))
ALLOCATE(x_spl2(Num_fit))
ALLOCATE(y_spl2(Num_fit))
ALLOCATE(x_spl4(Num_fit))
ALLOCATE(y_spl4(Num_fit))
ALLOCATE(w2(Num_fit))
!skip_col=0
coll_num=0
!write(*,*) "Begin Coll points"
i=1
Do
   summed_pix=0
   same_wave=0
   tmp_mean=0.
   tmp_w=0.
   tmp_err=0.
!   Do j=1,num_pix
   Do j=1,pix_run
    IF (i+j-1.le.NData) THEN
     IF (xin(i).eq.xin(i+j-1)) THEN
      same_wave=same_wave+1
      IF (nint(yin(i+j-1)).ne.-666) THEN
         summed_pix=summed_pix+1
!         tmp_mean=yin(i+j-1)/ein(i+j-1)**2+tmp_mean
!         tmp_w=1./ein(i+j-1)**2+tmp_w
         tmp_mean=yin(i+j-1)+tmp_mean
         tmp_w=1.+tmp_w
         tmp_err=tmp_err+ein(i+j-1)**2
!         rms_hold(summed_pix)=yin(i+j-1)
      END IF
     else
       EXIT
     END IF
    END IF
   End Do
   tmp_mean=tmp_mean/tmp_w
   IF (summed_pix.gt.1) THEN
   tmp_err=sqrt(tmp_err)/real(summed_pix-1.)
!         Call calc_stats(summed_pix,rms_hold,res_rms,res_mean)
         coll_num=coll_num+1
         x_spl2(coll_num)=xin(i)
         y_spl2(coll_num)=tmp_mean
         w2(coll_num)=1./(0.7+0.005*tmp_err)**2
!         w2(coll_num)=1./tmp_err**2
!         x_spl(coll_num)=xin(i)
!         y_spl(coll_num)=res_mean
!         w(int(i/num_pix)-skip_col)=1./res_rms
!         w(int(i/num_pix)-skip_col)=1.
!         w(coll_num)=1.0/abs(y_spl(coll_num))
!         w(coll_num)=1./res_rms
   ELSE IF (summed_pix.eq.1) THEN
         coll_num=coll_num+1
!         x_spl(coll_num)=xin(i)
!         y_spl(coll_num)=yin(i)
!         w(coll_num)=1.0/sqrt(abs(y_spl(coll_num)))
         x_spl2(coll_num)=xin(i)
         y_spl2(coll_num)=yin(i)
         w2(coll_num)=1./ein(i)**2
!         if (abs(y_spl(coll_num)).gt.0.00001) then
!            w2(coll_num)=1.0/sqrt(abs(y_spl(coll_num)))
!         else
!            w2(coll_num)=0.000001
!         end if
!   ELSE
!         skip_col=skip_col+1
   END IF
   i=i+same_wave
   IF (i.gt.NData) EXIT
End Do
!Num_fit=Num_fit-skip_col
Num_fit=coll_num
num_repeat=0
!i=1
!k=1
!write(*,*) "Begin remove redund"
!Step to remove points at same wavelength, which bspline can't handle
!multi_flag=0
!spec_tot=0.0
!Do
!   IF (k.gt.Num_fit) EXIT
!   IF (multi_flag.eq.0) THEN
!      tmp_mean=y_spl(k)
!      tmp_err=w(k)
!   END IF
!   x_spl2(i)=x_spl(k)
!   y_spl2(i)=tmp_mean
!!   spec_tot=spec_tot+y_spl2(i)
!   w2(i)=tmp_err
!!   w2(i)=1.0/sqrt(y_spl2(i))
!   multi_flag=0
!   j=0
!     IF (k+1.le.Num_fit) THEN
!      IF (x_spl(k).eq.x_spl(k+1)) THEN
!         multi_flag=1
!         j=1
!         tmp_mean=w(k)*y_spl(k)
!         tmp_err=(1./w(k))**2
!!         tmp_err=w(k)
!         tmp_w=w(k)
!         Do
!            IF (k+j.gt.Num_fit-1) EXIT
!            IF (x_spl(k+j).ne.x_spl(k+j+1)) EXIT
!            tmp_mean=tmp_mean+w(k+j)*y_spl(k+j)
!!            tmp_err=tmp_err+1./w(k+j)
!            tmp_err=tmp_err+(1./w(k+j))**2
!            tmp_w=tmp_w+w(k+j)
!            j=j+1
!         End Do
!      tmp_mean=tmp_mean/tmp_w
!!      tmp_err=1./w(k)
!      tmp_err=1./sqrt(tmp_err)
!      END IF
!     END IF
!   num_repeat=i
!   i=i+1
!   k=k+1+j
!End Do
!Num_fit=num_repeat
!open(16,file="dataout.txt")
!Do i=1,Num_fit
!   write(16,*) x_spl2(i),y_spl2(i), 1./w2(i)
!End Do
!Close(16)
!Here be oversampled output well suited to displaying the spline form for debugging
!ALLOCATE(x_spl3(Num_fit*10))
!ALLOCATE(y_spl3(Num_fit*10))
!Do i=1,Num_fit-1
!   Do j=1,10
!      x_spl3((i-1)*10+j)=x_spl2(i)+real(j-1)*(x_spl2(i+1)-x_spl2(i))/(10.0)
!   End Do
!End Do
!i=Num_fit
!Do j=1,10
!   x_spl3((i-1)*10+j)=x_spl2(i)+real(j-1)*(x_spl2(i)-x_spl2(i-1))/(10.0)
!End Do

!Need to cut out extreme points prior to spline fit
!open(67,file="cutlook.txt")
!ALLOCATE(w3(Num_fit))
!NData2=0
!Do i=1,Num_fit
!   Do j=max(1,i-cut_win),min(Num_fit,i+cut_win)
!      y_spl(j-max(1,j-cut_win)+1)=y_spl2(j)
!   End Do
!   call calc_stats(min(Num_fit,i+cut_win)-max(1,i-cut_win)+1,y_spl,res_rms,res_mean)
!write(67,*) x_spl2(i),abs(y_spl2(i)-res_mean),cut_val*res_rms
!   IF (abs(y_spl2(i)-res_mean).lt.cut_val*res_rms) THEN
!write(67,*) x_spl2(i),"Yes"
!      NData2=NData2+1
!      x_spl4(NData2)=x_spl2(i)
!      y_spl4(NData2)=y_spl2(i)
!      w3(NData2)=1.0
!   End If
!End Do
!close(67)
!Num_fit=NData2
!open(16,file="dataout.txt")
!Do i=1,Num_fit
!   write(16,*) x_spl4(i),y_spl4(i), w3(i)
!End Do
!Close(16)
!write(*,*) "rep",num_repeat
!nest=Num_fit+k_spline+1
!lwrk=Num_fit*(k_spline+1)+nest*(7+3*k_spline)
lwrk=Num_fit*(k_spline+1)+nest*(7+3*k_spline)
ALLOCATE(wrk(lwrk))
!s_val=spec_tot/s_mult
!s_val=Num_fit+sqrt(2.*Num_fit)*s_mult
!write(*,*) "Act s", s_val
!write(*,*) "Old s", (Num_fit-sqrt(2.*Num_fit))/2.0
!s_val=(Num_fit-sqrt(2.*Num_fit))/s_mult
!s_val=(Num_fit)/s_mult
!write(*,*) Num_fit, s_val
s_val=real(Num_fit)+s_mult*sqrt(2.*Num_fit)
!s_val=(Num_fit-sqrt(2.*Num_fit))/3.0
!s_val=Num_fit+sqrt(2.*Num_fit)*s_mult

xb=x_spl2(1)
xe=x_spl2(Num_fit)
!xb=x_spl4(1)
!xe=x_spl4(Num_fit)

!write(*,*) x_spl2(nint(Num_fit/2.)),x_spl2(Num_fit)
!write(*,*) y_spl2(nint(Num_fit/2.)),y_spl2(Num_fit)
!write(*,*) w2(nint(Num_fit/2.)),w2(Num_fit)
!write(*,*) iopt,k_spline,s_val,Num_fit,nest,lwrk
!write(*,*) "iopt,k_spline,s_val,Num_fit,nest,lwrk"
!write(*,*) xb,xe
!write(*,*) "Entering curfit"
!open(16,file="dataout.txt")
!Do i=1,Num_fit
!   write(16,*) x_spl2(i),y_spl2(i)
!End Do
!close(16)
!read(*,*) iopt
ier=0

!write(*,*) "Here are the checks"
!write(*,*) "1,iopt",iopt
!write(*,*) "2,k",k_spline
!write(*,*) "3,m",Num_fit
!write(*,*) "4,nest and mod k and 8",nest,2.0*k_spline+2.0,Num_fit+k_spline+1
!err1=0
!Do i=1,Num_fit
!   IF (w2(i).le.0.0) THEN
!      write(*,*) x_spl2(i),y_spl2(i),w2(i)
!      err1=1
!   END IF
!End Do
!write(*,*) "5,w cond",err1
!err2=0
!Do i=2,Num_fit
!   IF (x_spl2(i-1).ge.x_spl2(i)) err2=1
!End Do
!write(*,*) "6,x cond",err2
!write(*,*) "7,lwrk",lwrk,(k_spline+1)*Num_fit+nest*(7+3*k_spline)
!knot_space=5
!skip_knot=0
!knots(1)=x_spl2(1)
!knots(2)=x_spl2(1)
!knots(3)=x_spl2(1)
!Do i=1,Num_fit,knot_space
!      knots(4+floor(real((i-1)/knot_space)))=x_spl2(i)
!End Do
!num_knots=3+floor(real(Num_fit/knot_space))
!knots(num_knots+1)=x_spl2(Num_fit)
!knots(num_knots+2)=x_spl2(Num_fit)
!knots(num_knots+3)=x_spl2(Num_fit)
!num_knots=num_knots+3
!JJA cut here 1
knots(1)=x_spl2(1) 
knots(2)=x_spl2(1) 
knots(3)=x_spl2(1)
num_knots=3
i=1
Do
   Do
      IF (x_spl2(i)-knots(num_knots).ge.min_knot_space) THEN
         EXIT
      End If
      i=i+1
      IF (i.ge.Num_fit) EXIT 
   End Do
   IF (i.ge.Num_fit) EXIT 
   knots(num_knots+1)=x_spl2(i)
   num_knots=num_knots+1
!   knots(num_knots+1)=knots(num_knots)+s_mult
!   IF (knots(num_knots+1).ge.x_spl2(Num_fit)) EXIT
!   num_knots=num_knots+1
End Do
knots(num_knots+1)=x_spl2(Num_fit)
knots(num_knots+2)=knots(num_knots+1)
knots(num_knots+3)=knots(num_knots+1)
num_knots=num_knots+3
!JJA cut here 2
call curfit(iopt,Num_fit,x_spl2,y_spl2,w2,xb,xe,k_spline,s_val,nest,num_knots&
,knots,coeffs,ls_fit,wrk,lwrk,iwrk,ier)
!call curfit(iopt,Num_fit,x_spl4,y_spl4,w3,xb,xe,k_spline,s_val,nest,num_knots&
!,knots,coeffs,ls_fit,wrk,lwrk,iwrk,ier)
!write(*,*) "Out"
!write(*,*) "Exiting curfit"
!write(*,*) ier
!write(*,*) "In2", ier
!write(*,*) "Out2"
!write(*,*) ier
!IF (fib_cur.eq.82) THEN
!open(65,file="myknots.txt")
!Do i=1,num_knots
!    write(65,*) knots(i),1.4
!End Do
!close(65)
!!END IF
!call splev(knots,num_knots,coeffs,k_spline,x_spl3,y_spl3,Num_fit*10,ier)
!IF (fib_cur.eq.6) THEN
!write(*,*) fib_cur
!open(15,file="splineout.txt")
!Do i=1,Num_fit*10
!   write(15,*) x_spl3(i),y_spl3(i)
!End Do
!Close(15)
!STOP
!END IF
!The problem is that the dierckx libraries let knots get too close together
!My solution is to use their automatic knot finding and then manually parse 
!the list for too-closely spaced knots. I then run the spline solution with these 
!set knots and go home.
!min_knot_space=0.75
!min_knot_space=0.35

!skip_knot=0
!Do i=buffer1,num_knots-buffer1
!   IF (knots(i)-knots(i-1-skip_knot).le.min_knot_space) THEN
!      skip_knot=skip_knot+1
!   ELSE
!      knots(i-skip_knot)=knots(i)
!   END IF 
!End Do
!num_knots=num_knots-skip_knot

!open(65,file="myknots.txt")
!Do i=1,num_knots
!    write(65,*) knots(i),1.4
!End Do
!close(65)

!iopt=-1
!call curfit(iopt,Num_fit,x_spl2,y_spl2,w2,xb,xe,k_spline,s_val,nest,num_knots&
!,knots,coeffs,ls_fit,wrk,lwrk,iwrk,ier)

!END IF
!call splev(knots,num_knots,coeffs,k_spline,x_spl3,y_spl3,Num_fit*10,ier)
!open(15,file="splineout2.txt")
!Do i=1,Num_fit*10
!   write(15,*) x_spl3(i),y_spl3(i)
!End Do
!Close(15)
!write(*,*) "Give me value"
!read(*,*) dummy1
!DEALLOCATE(x_spl3)
!DEALLOCATE(y_spl3)
!DEALLOCATE(w3)
DEALLOCATE(rms_hold)
DEALLOCATE(w)
DEALLOCATE(w2)
DEALLOCATE(wrk)
DEALLOCATE(x_spl)
DEALLOCATE(y_spl)
DEALLOCATE(x_spl2)
DEALLOCATE(y_spl2)
DEALLOCATE(x_spl4)
DEALLOCATE(y_spl4)
RETURN
END Subroutine
!------------------------------------------------------
Subroutine calc_stats(num_pts,in_array,calc_rms,calc_mean)
IMPLICIT NONE
REAL(KIND=4)::calc_rms,sum1,sum2,in_array(num_pts),calc_mean,&
select
Integer::num_pts,i
!call biwgt(in_array,num_pts,calc_mean,calc_rms)
!calc_rms=0.0
sum1=0.
sum2=0.
Do i=1,num_pts
   sum1=sum1+in_array(i)
   sum2=sum2+in_array(i)**2
End Do
calc_rms=sqrt((sum2-sum1**2/num_pts)/real(num_pts-1))
calc_mean=sum1/num_pts
!Nope, doing median now
!calc_mean=select(nint((num_pts+1)/2.),num_pts,in_array)
RETURN
End Subroutine
