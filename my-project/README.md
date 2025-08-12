# 🚀 Ad-Pod Stitching Server Optimization

A production-grade full-stack Python application that solves the **Ad-Pod Stitching Server Optimization** problem using the **CFLP-Heuristic** (Clustering → Reduced Capacitated Facility Location Problem → Exact MILP).

## 📋 Problem Overview

The application optimizes the placement of Ad-Pod Stitching Servers for video-on-demand streaming across the United States. Each server has:
- Fixed setup cost (activation)
- Capacity limit (max concurrent streams)

Each Designated Market Area (DMA) has:
- Geographic location (x, y coordinates)
- Demand for concurrent streams

**Goal**: Minimize total cost (setup + delivery) by selecting servers to activate and assigning every DMA to exactly one activated server, while respecting server capacities.

## 🏗️ Architecture

- **Backend**: FastAPI with CFLP-Heuristic optimization engine
- **Frontend**: Streamlit UI for data upload and results visualization
- **Optimization**: K-means clustering + MILP solving with PuLP
- **Deployment**: Docker containers with health checks and resource limits

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd my-project

# Generate sample data
python scripts/generate_sample_data.py

# Start services
docker-compose up --build

# Access the application
# Frontend: http://localhost:8501
# Backend API: http://localhost:8000
```

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Generate sample data
python scripts/generate_sample_data.py

# Start backend (Terminal 1 from project root)
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000

# Start frontend (Terminal 2 from project root)
streamlit run frontend/app.py --server.port 8501
```

## 📁 Project Structure

```
my-project/
├── backend/                 # FastAPI backend
│   ├── app.py             # Main API application
│   ├── optimization.py    # CFLP-Heuristic engine
│   ├── models.py          # Pydantic data models
│   └── utils.py           # Utility functions
├── frontend/               # Streamlit frontend
│   └── app.py             # Main UI application
├── data/                   # Sample data and outputs
│   ├── servers_sample.csv # Sample server data
│   └── dmas_sample.csv    # Sample DMA data
├── scripts/                # Utility scripts
│   └── generate_sample_data.py
├── tests/                  # Test suite
├── Dockerfile              # Backend container
├── docker-compose.yml      # Multi-service orchestration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Configuration

### Backend Configuration

- **Port**: 8000 (configurable)
- **MILP Timeout**: 25 seconds (configurable)
- **Random Seed**: 42 (for reproducibility)
- **CORS**: Enabled for localhost:8501

### Frontend Configuration

- **Port**: 8501 (configurable)
- **Backend URL**: http://localhost:8000 (configurable)
- **File Upload**: CSV format only
- **Real-time Updates**: Automatic refresh on optimization completion

## 📊 Data Format

### Servers CSV (`servers.csv`)
```csv
server_id,location_x,location_y,setup_cost,capacity_streams
S001,25.5,67.2,45.0,300
S002,78.1,34.8,67.5,450
...
```

**Columns:**
- `server_id`: Unique server identifier (string)
- `location_x`: X coordinate (float, -180 to 180)
- `location_y`: Y coordinate (float, -90 to 90)
- `setup_cost`: Server activation cost (float > 0)
- `capacity_streams`: Maximum concurrent streams (integer > 0)

### DMAs CSV (`dmas.csv`)
```csv
dma_id,location_x,location_y,demand_streams
D001,30.2,65.8,5
D002,75.4,40.1,8
...
```

**Columns:**
- `dma_id`: Unique DMA identifier (string)
- `location_x`: X coordinate (float, -180 to 180)
- `location_y`: Y coordinate (float, -90 to 90)
- `demand_streams`: Required concurrent streams (integer ≥ 0)

## 🎯 API Endpoints

### Backend API (FastAPI)

- **GET** `/` - API information and available endpoints
- **GET** `/health` - Health check endpoint
- **POST** `/optimize` - Main optimization endpoint

#### Optimization Request
```bash
POST /optimize
Content-Type: multipart/form-data

servers: <servers.csv file>
dmas: <dmas.csv file>
```

#### Optimization Response
```json
{
  "activated_servers": ["S001", "S003"],
  "assignments": {
    "D001": "S001",
    "D002": "S001",
    "D003": "S003"
  },
  "costs": {
    "total_setup_cost": 112.5,
    "total_delivery_cost": 45.23,
    "total_cost": 157.73
  },
  "metadata": {
    "execution_time": 12.45,
    "approach_used": "clustering",
    "k_used": 8
  }
}
```

## 🔬 Algorithm Details

### CFLP-Heuristic Approach

1. **Data Validation**: Check CSV format and feasibility
2. **Distance Calculation**: Compute Euclidean distances between servers and DMAs
3. **K-Range Determination**: Calculate feasible clustering range
4. **Iterative Clustering**: For each K value:
   - Perform K-means clustering on DMA locations
   - Aggregate cluster demands and distances
   - Formulate reduced CFLP as MILP
   - Solve with PuLP (CBC solver)
   - Recover DMA-level assignments
5. **Solution Selection**: Choose best solution by total cost
6. **Fallback**: Use greedy algorithm if MILP fails

### Performance Characteristics

- **Scalability**: Up to 1,000 servers and 50,000 DMAs
- **Performance**: <30 seconds end-to-end for max scale
- **Memory**: Optimized with float32 distance matrices
- **Reproducibility**: Fixed random seeds for clustering

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov=frontend

# Run specific test file
pytest tests/test_backend.py
```

### Test Coverage
- CSV validation and parsing
- Optimization algorithm correctness
- API endpoint functionality
- Error handling and edge cases
- Performance benchmarks

## 🐳 Docker Deployment

### Production Deployment
```bash
# Build and start services
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Scale backend (if needed)
docker-compose up --scale backend=3 -d
```

### Health Checks
- Backend: HTTP endpoint `/health`
- Frontend: Streamlit service status
- Automatic restart on failure
- Resource limits and reservations

## 📈 Performance Optimization

### Backend Optimizations
- **Vectorized Operations**: NumPy for distance calculations
- **Memory Management**: float32 for large matrices
- **Parallel Processing**: ThreadPoolExecutor for CPU-intensive tasks
- **Timeout Handling**: Configurable MILP solver timeouts

### Frontend Optimizations
- **Caching**: Streamlit caching for data operations
- **Async Operations**: Non-blocking API calls
- **Progressive Loading**: Spinner states and progress indicators

## 🔒 Security Features

- **Input Validation**: Pydantic models for data validation
- **File Upload Security**: CSV format validation
- **CORS Configuration**: Restricted to trusted origins
- **Non-root Containers**: Docker security best practices

## 🚨 Error Handling

### Common Error Scenarios
- **Invalid CSV Format**: Clear error messages with column requirements
- **Infeasible Problems**: Demand > Capacity scenarios
- **Optimization Timeouts**: Configurable fallback to greedy algorithm
- **Network Issues**: Connection error handling and retry logic

### Error Response Format
```json
{
  "error": "Data validation failed",
  "error_code": "HTTP_400",
  "detail": "Missing required server columns: ['setup_cost']"
}
```

## 📚 API Documentation

### Interactive API Docs
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 🔧 Troubleshooting

### Common Issues

#### Backend Won't Start
```bash
# Check port availability
netstat -an | grep 8000

# Check Python dependencies
pip list | grep fastapi

# Check logs
docker-compose logs backend
```

#### Frontend Connection Issues
```bash
# Verify backend is running
curl http://localhost:8000/health

# Check CORS configuration
# Ensure backend allows localhost:8501
```

#### Optimization Fails
```bash
# Check data feasibility
python scripts/generate_sample_data.py --validate

# Reduce MILP timeout
# Adjust timeout_seconds parameter
```

### Performance Issues
- **Large Datasets**: Use smaller K values for clustering
- **Memory Issues**: Reduce dataset size or use float32
- **Slow Optimization**: Increase MILP timeout or use greedy fallback

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure PEP-8 compliance
5. Submit a pull request

### Code Standards
- **Type Hints**: Required for all functions
- **Docstrings**: Google-style documentation
- **Testing**: Minimum 80% coverage
- **Formatting**: Black code formatter

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **FastAPI**: Modern, fast web framework
- **Streamlit**: Rapid web app development
- **PuLP**: Linear programming solver
- **SciPy/NumPy**: Scientific computing libraries
- **scikit-learn**: Machine learning algorithms

## 📞 Support

For questions, issues, or contributions:
- **Issues**: GitHub issue tracker
- **Documentation**: API docs at `/docs`
- **Community**: Project discussions

---

**Built with ❤️ for production-grade optimization solutions**
