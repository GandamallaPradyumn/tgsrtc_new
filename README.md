# 🚍 TGSRTC Driver & Depot Dashboard  
### A Streamlit + MySQL based Productivity & Operations Monitoring System

![TGSRTC Logo](LOGO.png)

This project is a complete Driver, Depot, Region, and Productivity Dashboard system designed for **Telangana State Road Transport Corporation (TGSRTC)**.  
It simplifies daily depot operations, driver performance tracking, absenteeism monitoring, and management reporting.

---

## ⭐ Key Features

### **1. Secure Login System**
- User authentication (DM, RM, Admin)  
- Role–based access  
- Depot / Region restrictions  
- Encrypted passwords  

### **2. Driver Dashboard**
Shows complete driver history:
- Daily KMs  
- Daily earnings  
- Services operated  
- Day/Night performance  
- Route details  
- Depot average vs driver performance  

### **3. Depot Dashboard (DM & RM)**
- Planned vs Actual services  
- Driver strength  
- Sick leave %  
- Spot absent %  
- Double duty %  
- Weekly off %  
- Real-time variance indicators  

### **4. Productivity 8 Ratios**
Available for:  
- Depot Manager (DM)  
- Regional Manager (RM)

Metrics include:  
- Weekly Off  
- Special Off  
- Sick Leave  
- Double Duty  
- Off Cancellation  
- Spot Absent  
- Long Leave/Absent  
- Drivers/Schedule Ratio  

Benchmarks change automatically for *Rural* and *Urban* depots.

### **5. LSA (Leave / Sick / Absent) Upload**
- Auto-cleaning of CSV  
- Missing value checks  
- Depot mapping  
- Loads into `driver_absenteeism` table

### **6. Daily Operations Upload**
- Cleans CSV  
- Maps depot names  
- Converts dates  
- Validates missing fields  
- Loads into `daily_operations` table  

### **7. Pending Depot Status Report**
Shows:
- Latest updated date  
- Days pending  
- Updated / Pending status  
- Zone → Region → Depot format  

### **8. Automated ETL**
Folder: `/ETL/`  
- Multi-file CSV loader  
- Data transformation  
- MySQL uploader  
- Validation rules  

---








## 🗂 Folder Structure

