import pandas as pd
import io
import random
from faker import Faker
from datetime import datetime, timedelta
from typing import List, Tuple, Any, IO
from generators.base import DataGenerator

class TabularGenerator(DataGenerator):
    """Generator for tabular data (CSV, XLSX)."""

    def __init__(self):
        self.fake = Faker()

    def generate(self, original_file: IO[bytes], num_files: int = 100, num_rows: int = None, file_ext: str = 'csv', **kwargs) -> List[Tuple[str, bytes]]:
        if file_ext == 'csv':
            df = pd.read_csv(original_file)
        elif file_ext == 'xlsx':
            df = pd.read_excel(original_file)
        else:
            raise ValueError("Unsupported tabular format")

        if num_rows is None:
            num_rows = len(df)

        generated_files = []

        for file_idx in range(num_files):
            synthetic_data = {}

            for col in df.columns:
                dtype = df[col].dtype

                if pd.api.types.is_integer_dtype(dtype):
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if pd.isna(min_val): min_val = 0
                    if pd.isna(max_val): max_val = 100
                    synthetic_data[col] = [random.randint(int(min_val), int(max_val)) for _ in range(num_rows)]

                elif pd.api.types.is_float_dtype(dtype):
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if pd.isna(min_val): min_val = 0.0
                    if pd.isna(max_val): max_val = 100.0
                    synthetic_data[col] = [random.uniform(min_val, max_val) for _ in range(num_rows)]

                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    start_date = df[col].min()
                    end_date = df[col].max()
                    if pd.isna(start_date): start_date = datetime.now() - timedelta(days=365)
                    if pd.isna(end_date): end_date = datetime.now()

                    time_between_dates = end_date - start_date
                    days_between_dates = time_between_dates.days

                    synthetic_data[col] = [
                        start_date + timedelta(days=random.randrange(days_between_dates + 1))
                        if days_between_dates > 0 else start_date
                        for _ in range(num_rows)
                    ]

                else:
                    col_lower = col.lower()
                    if 'email' in col_lower:
                        synthetic_data[col] = [self.fake.email() for _ in range(num_rows)]
                    elif 'name' in col_lower:
                        synthetic_data[col] = [self.fake.name() for _ in range(num_rows)]
                    elif 'address' in col_lower:
                        synthetic_data[col] = [self.fake.address() for _ in range(num_rows)]
                    elif 'date' in col_lower:
                        synthetic_data[col] = [self.fake.date() for _ in range(num_rows)]
                    else:
                        synthetic_data[col] = [self.fake.word() for _ in range(num_rows)]

            result_df = pd.DataFrame(synthetic_data)

            if file_ext == 'csv':
                data = result_df.to_csv(index=False).encode('utf-8')
            else:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    result_df.to_excel(writer, index=False)
                data = output.getvalue()

            generated_files.append((f"generated_{file_idx+1}_{original_file.name}", data))

        return generated_files
