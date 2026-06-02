import pandas as pd
import hashlib
import logging
import os

# ─────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger()

# ─────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────
log.info("Loading data files...")

df1 = pd.read_excel("../Data/retail_data1.xlsx")
df2 = pd.read_excel("../Data/retail_data2.xlsx")
product_dim = pd.read_excel("../Data/product_details.xlsx")

log.info(f"retail_data1: {df1.shape[0]} rows")
log.info(f"retail_data2: {df2.shape[0]} rows")

# ─────────────────────────────────────────
# STEP 2 — COMBINE BOTH DATASETS
# ─────────────────────────────────────────
log.info("Combining datasets...")

df = pd.concat([df1, df2], ignore_index=True)
log.info(f"Combined total rows: {df.shape[0]}")

# ─────────────────────────────────────────
# STEP 3 — REMOVE DUPLICATES
# ─────────────────────────────────────────
log.info("Removing duplicates...")

before = df.shape[0]

# Sort so 'successful' payments come first
df = df.sort_values('payment_status', ascending=True)

# Drop duplicates keeping first (successful) record
df = df.drop_duplicates(
    subset=['transaction_id', 'customer_id', 'product_id', 'transaction_date'],
    keep='first'
)

after = df.shape[0]
log.info(f"Removed {before - after} duplicate rows. Remaining: {after}")

# ─────────────────────────────────────────
# STEP 4 — FIX DATE FORMATS
# ─────────────────────────────────────────
log.info("Fixing date formats...")

df['transaction_date'] = pd.to_datetime(
    df['transaction_date'],
    dayfirst=False,
    errors='coerce'
)

bad_dates = df['transaction_date'].isna().sum()
log.info(f"Rows with invalid dates: {bad_dates}")

# Drop rows where date could not be parsed
df = df.dropna(subset=['transaction_date'])

# ─────────────────────────────────────────
# STEP 5 — FIX MISSING PRICES
# ─────────────────────────────────────────
log.info("Fixing missing prices...")

# Build a price lookup from product_dim
price_map = product_dim.set_index('product_id')['price'].to_dict()

# Fill missing prices using product_id lookup
df['price'] = df.apply(
    lambda row: price_map.get(row['product_id'], row['price'])
    if pd.isna(row['price']) else row['price'],
    axis=1
)

missing_prices = df['price'].isna().sum()
log.info(f"Rows still missing price after fix: {missing_prices}")

# Drop rows where price is still missing
df = df.dropna(subset=['price'])

# ─────────────────────────────────────────
# STEP 6 — STANDARDIZE TEXT COLUMNS
# ─────────────────────────────────────────
log.info("Standardizing text columns...")

# Fix Category names
category_map = {
    'elec'            : 'Electronics',
    'electronics'     : 'Electronics',
    'furn'            : 'Furniture',
    'furniture'       : 'Furniture',
    'cloth'           : 'Clothing',
    'clothing'        : 'Clothing',
    'home appliances' : 'Home Appliances',
    'home'            : 'Home Appliances',
}

df['category'] = (
    df['category']
    .str.lower()
    .str.strip()
    .map(lambda x: category_map.get(x, x))
)

# Fix Product Names → Title Case
df['product_name'] = df['product_name'].str.strip().str.title()

# Fix City → Title Case
df['city'] = df['city'].str.strip().str.title()

# Fix purchase_location → lowercase
df['purchase_location'] = df['purchase_location'].str.lower().str.strip()

# Fix payment_method → Title Case
df['payment_method'] = df['payment_method'].str.strip().str.title()

# Fix payment_status → lowercase
df['payment_status'] = df['payment_status'].str.lower().str.strip()

log.info("Text columns standardized.")

# ─────────────────────────────────────────
# STEP 7 — REMOVE INVALID QUANTITIES
# ─────────────────────────────────────────
log.info("Removing invalid quantities...")

before = df.shape[0]
df = df[df['quantity'] > 0]
log.info(f"Removed {before - df.shape[0]} rows with invalid quantity.")

# ─────────────────────────────────────────
# STEP 8 — KEEP ONLY SUCCESSFUL PAYMENTS
# ─────────────────────────────────────────
log.info("Filtering only successful transactions...")

before = df.shape[0]
df = df[df['payment_status'] == 'successful']
log.info(f"Removed {before - df.shape[0]} failed/pending transactions.")

# ─────────────────────────────────────────
# STEP 9 — MASK PII (Email & Phone)
# ─────────────────────────────────────────
log.info("Masking PII data...")

def mask_email(email):
    if pd.isna(email):
        return email
    parts = str(email).split('@')
    if len(parts) == 2:
        return parts[0][:2] + '****@' + parts[1]
    return '****'

def hash_phone(phone):
    if pd.isna(phone):
        return phone
    return hashlib.sha256(str(phone).encode()).hexdigest()[:10]

df['email'] = df['email'].apply(mask_email)
df['phone'] = df['phone'].apply(hash_phone)

log.info("PII masking done.")

# ─────────────────────────────────────────
# STEP 10 — CALCULATE REVENUE
# ─────────────────────────────────────────
log.info("Calculating revenue...")

# Revenue = price × quantity × (1 - discount)
df['revenue'] = df['price'] * df['quantity'] * (1 - df['discount'])
df['revenue'] = df['revenue'].round(2)

log.info("Revenue column created.")

# ─────────────────────────────────────────
# STEP 11 — ADD EXTRA DATE COLUMNS
# ─────────────────────────────────────────
log.info("Adding date columns...")

df['year']  = df['transaction_date'].dt.year
df['month'] = df['transaction_date'].dt.month
df['month_name'] = df['transaction_date'].dt.strftime('%B')  # January, February...
df['quarter'] = df['transaction_date'].dt.quarter

# ─────────────────────────────────────────
# STEP 12 — CALCULATE KPIs
# ─────────────────────────────────────────
log.info("Calculating KPIs...")

total_revenue        = df['revenue'].sum()
total_orders         = df['transaction_id'].nunique()
avg_order_value      = round(total_revenue / total_orders, 2)
total_units_sold     = df['quantity'].sum()

revenue_by_category  = df.groupby('category')['revenue'].sum().reset_index()
revenue_by_city      = df.groupby('city')['revenue'].sum().reset_index()
revenue_by_month     = df.groupby(['year','month','month_name'])['revenue'].sum().reset_index()
revenue_by_channel   = df.groupby('purchase_location')['revenue'].sum().reset_index()
revenue_by_payment   = df.groupby('payment_method')['revenue'].sum().reset_index()
top_products         = df.groupby('product_name')['revenue'].sum().sort_values(ascending=False).reset_index()

print("\n" + "="*50)
print("         BUSINESS KPI SUMMARY")
print("="*50)
print(f"  Total Revenue      : ₹{total_revenue:,.2f}")
print(f"  Total Orders       : {total_orders}")
print(f"  Avg Order Value    : ₹{avg_order_value:,.2f}")
print(f"  Total Units Sold   : {total_units_sold}")
print("="*50)

print("\n📦 Revenue by Category:")
print(revenue_by_category.to_string(index=False))

print("\n🏙️ Revenue by City:")
print(revenue_by_city.to_string(index=False))

print("\n🛒 Revenue by Channel:")
print(revenue_by_channel.to_string(index=False))

print("\n💳 Revenue by Payment Method:")
print(revenue_by_payment.to_string(index=False))

print("\n🏆 Top Products by Revenue:")
print(top_products.to_string(index=False))

# ─────────────────────────────────────────
# STEP 13 — EXPORT CLEANED DATA
# ─────────────────────────────────────────
log.info("Exporting cleaned data...")

output_path = "../Data/cleaned_retail_data.xlsx"
df.to_excel(output_path, index=False)

log.info(f"✅ Cleaned data saved to: {output_path}")
log.info(f"✅ Final dataset has {df.shape[0]} rows and {df.shape[1]} columns")

print("\n✅ Pipeline Complete! Cleaned file saved to Data folder.")