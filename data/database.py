"""
Database operations for the E-Commerce Price Prediction System
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime, timedelta
import logging

from config.config import DATABASE_CONFIG

class DatabaseManager:
    """Manages all database operations for the e-commerce system"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DATABASE_CONFIG['database_path']
        self.ensure_database_exists()
        self.create_tables()
    
    def ensure_database_exists(self):
        """Ensure the database directory exists"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def create_tables(self):
        """Create all necessary tables"""
        with self.get_connection() as conn:
            # Products table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    brand TEXT,
                    price REAL NOT NULL,
                    description TEXT,
                    features TEXT,  -- JSON string
                    rating REAL DEFAULT 0,
                    num_reviews INTEGER DEFAULT 0,
                    availability INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Users table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    age INTEGER,
                    gender TEXT,
                    preferences TEXT,  -- JSON string
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Purchases/Interactions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    product_id INTEGER NOT NULL,
                    interaction_type TEXT NOT NULL,  -- 'purchase', 'view', 'rating', 'cart'
                    rating REAL,
                    quantity INTEGER DEFAULT 1,
                    price_paid REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    FOREIGN KEY (product_id) REFERENCES products (id)
                )
            ''')
            
            # Categories table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS categories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    parent_category TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Price history table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id INTEGER NOT NULL,
                    price REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (product_id) REFERENCES products (id)
                )
            ''')
            
            # Model performance table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,  -- 'price_prediction', 'recommendation'
                    accuracy_metric REAL,
                    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    parameters TEXT,  -- JSON string
                    performance_data TEXT  -- JSON string
                )
            ''')
            
            conn.commit()
    
    # Product operations
    def add_product(self, product_data: Dict[str, Any]) -> int:
        """Add a new product to the database"""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                INSERT INTO products (name, category, brand, price, description, features, rating, num_reviews)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                product_data['name'],
                product_data['category'],
                product_data.get('brand'),
                product_data['price'],
                product_data.get('description'),
                json.dumps(product_data.get('features', {})),
                product_data.get('rating', 0),
                product_data.get('num_reviews', 0)
            ))
            return cursor.lastrowid
    
    def get_products(self, category: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """Get products from database"""
        query = "SELECT * FROM products WHERE availability = 1"
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def update_product(self, product_id: int, updates: Dict[str, Any]) -> bool:
        """Update product information"""
        if not updates:
            return False
        
        set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
        values = list(updates.values())
        values.append(product_id)
        
        with self.get_connection() as conn:
            cursor = conn.execute(f'''
                UPDATE products SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', values)
            return cursor.rowcount > 0
    
    def delete_product(self, product_id: int) -> bool:
        """Soft delete a product"""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                UPDATE products SET availability = 0, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (product_id,))
            return cursor.rowcount > 0
    
    # User operations
    def add_user(self, user_data: Dict[str, Any]) -> int:
        """Add a new user to the database"""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                INSERT INTO users (username, email, age, gender, preferences)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                user_data['username'],
                user_data['email'],
                user_data.get('age'),
                user_data.get('gender'),
                json.dumps(user_data.get('preferences', {}))
            ))
            return cursor.lastrowid
    
    def get_users(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Get users from database"""
        query = "SELECT * FROM users"
        if limit:
            query += f" LIMIT {limit}"
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific user by ID"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    # Interaction operations
    def add_interaction(self, interaction_data: Dict[str, Any]) -> int:
        """Add a user-product interaction"""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                INSERT INTO interactions (user_id, product_id, interaction_type, rating, quantity, price_paid)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                interaction_data['user_id'],
                interaction_data['product_id'],
                interaction_data['interaction_type'],
                interaction_data.get('rating'),
                interaction_data.get('quantity', 1),
                interaction_data.get('price_paid')
            ))
            return cursor.lastrowid
    
    def get_interactions(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Get all interactions from database"""
        query = '''
            SELECT i.*, p.name as product_name, p.category, p.brand, u.username
            FROM interactions i
            JOIN products p ON i.product_id = p.id
            JOIN users u ON i.user_id = u.id
            ORDER BY i.timestamp DESC
        '''
        
        if limit:
            query += f" LIMIT {limit}"
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)
    
    def get_user_interactions(self, user_id: int, interaction_type: Optional[str] = None) -> pd.DataFrame:
        """Get interactions for a specific user"""
        query = '''
            SELECT i.*, p.name as product_name, p.category, p.brand
            FROM interactions i
            JOIN products p ON i.product_id = p.id
            WHERE i.user_id = ?
        '''
        params = [user_id]
        
        if interaction_type:
            query += " AND i.interaction_type = ?"
            params.append(interaction_type)
        
        query += " ORDER BY i.timestamp DESC"
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_product_interactions(self, product_id: int) -> pd.DataFrame:
        """Get interactions for a specific product"""
        query = '''
            SELECT i.*, u.username
            FROM interactions i
            JOIN users u ON i.user_id = u.id
            WHERE i.product_id = ?
            ORDER BY i.timestamp DESC
        '''
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=[product_id])
    
    # Analytics methods
    def get_sales_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get sales analytics for the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with self.get_connection() as conn:
            # Total sales
            total_sales = conn.execute('''
                SELECT COUNT(*) as count, SUM(price_paid) as revenue
                FROM interactions
                WHERE interaction_type = 'purchase' AND timestamp >= ?
            ''', (cutoff_date,)).fetchone()
            
            # Sales by category
            category_sales = pd.read_sql_query('''
                SELECT p.category, COUNT(*) as sales_count, SUM(i.price_paid) as revenue
                FROM interactions i
                JOIN products p ON i.product_id = p.id
                WHERE i.interaction_type = 'purchase' AND i.timestamp >= ?
                GROUP BY p.category
                ORDER BY revenue DESC
            ''', conn, params=[cutoff_date])
            
            # Top products
            top_products = pd.read_sql_query('''
                SELECT p.name, p.category, COUNT(*) as sales_count, SUM(i.price_paid) as revenue
                FROM interactions i
                JOIN products p ON i.product_id = p.id
                WHERE i.interaction_type = 'purchase' AND i.timestamp >= ?
                GROUP BY p.id, p.name, p.category
                ORDER BY sales_count DESC
                LIMIT 10
            ''', conn, params=[cutoff_date])
            
            return {
                'total_sales': dict(total_sales) if total_sales else {'count': 0, 'revenue': 0},
                'category_sales': category_sales,
                'top_products': top_products
            }
    
    def get_user_item_matrix(self) -> pd.DataFrame:
        """Get user-item interaction matrix for collaborative filtering"""
        query = '''
            SELECT user_id, product_id, 
                   CASE 
                       WHEN interaction_type = 'purchase' THEN 5
                       WHEN interaction_type = 'cart' THEN 3
                       WHEN interaction_type = 'view' THEN 1
                       WHEN rating IS NOT NULL THEN rating
                       ELSE 1
                   END as score
            FROM interactions
        '''
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn)
            return df.pivot_table(index='user_id', columns='product_id', values='score', fill_value=0)
    
    # Price history operations
    def add_price_history(self, product_id: int, price: float, timestamp: Optional[str] = None) -> int:
        """Add a price history entry"""
        with self.get_connection() as conn:
            if timestamp:
                cursor = conn.execute('''
                    INSERT INTO price_history (product_id, price, timestamp)
                    VALUES (?, ?, ?)
                ''', (product_id, price, timestamp))
            else:
                cursor = conn.execute('''
                    INSERT INTO price_history (product_id, price)
                    VALUES (?, ?)
                ''', (product_id, price))
            return cursor.lastrowid
    
    def get_price_history(self, product_id: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """Get price history for products"""
        query = '''
            SELECT ph.*, p.name as product_name, p.category, p.brand
            FROM price_history ph
            JOIN products p ON ph.product_id = p.id
        '''
        params = []
        
        if product_id:
            query += " WHERE ph.product_id = ?"
            params.append(product_id)
        
        query += " ORDER BY ph.timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def update_product_price(self, product_id: int, new_price: float) -> bool:
        """Update product price and add to price history"""
        with self.get_connection() as conn:
            # Get current price
            current_price = conn.execute('''
                SELECT price FROM products WHERE id = ?
            ''', (product_id,)).fetchone()
            
            if current_price and current_price[0] != new_price:
                # Add to price history
                self.add_price_history(product_id, new_price)
                
                # Update product price
                cursor = conn.execute('''
                    UPDATE products SET price = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (new_price, product_id))
                return cursor.rowcount > 0
            return False
    
    def get_price_trends(self, days: int = 30) -> pd.DataFrame:
        """Get price trends for the last N days"""
        query = '''
            SELECT 
                DATE(ph.timestamp) as date,
                p.category,
                AVG(ph.price) as avg_price,
                MIN(ph.price) as min_price,
                MAX(ph.price) as max_price,
                COUNT(*) as price_changes
            FROM price_history ph
            JOIN products p ON ph.product_id = p.id
            WHERE ph.timestamp >= datetime('now', '-{} days')
            GROUP BY DATE(ph.timestamp), p.category
            ORDER BY date DESC, category
        '''.format(days)
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)
    
    # Utility methods
    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """Create a backup of the database"""
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = DATABASE_CONFIG['backup_path']
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"backup_{timestamp}.db"
        
        # Simple file copy for SQLite
        import shutil
        shutil.copy2(self.db_path, backup_path)
        return str(backup_path)
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get basic database statistics"""
        with self.get_connection() as conn:
            stats = {}
            tables = ['products', 'users', 'interactions', 'categories']
            
            for table in tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[table] = count
            
            return stats
    
    def clear_all_data(self) -> bool:
        """Clear all data from the database (for testing/reset)"""
        with self.get_connection() as conn:
            tables = ['interactions', 'price_history', 'model_performance', 'products', 'users', 'categories']
            
            for table in tables:
                conn.execute(f"DELETE FROM {table}")
            
            conn.commit()
            return True