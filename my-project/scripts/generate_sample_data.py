#!/usr/bin/env python3
"""
Sample data generator for the Ad-Pod Stitching Server Optimization application.

This script generates realistic sample CSV files for testing the optimization
algorithm with configurable dataset sizes and geographic distributions.
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_servers_data(
    num_servers: int,
    x_range: Tuple[float, float] = (0, 100),
    y_range: Tuple[float, float] = (0, 100),
    setup_cost_range: Tuple[float, float] = (20, 100),
    capacity_range: Tuple[int, int] = (100, 500),
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate sample server data.
    
    Args:
        num_servers: Number of servers to generate
        x_range: Range for X coordinates
        y_range: Range for Y coordinates
        setup_cost_range: Range for setup costs
        capacity_range: Range for server capacities
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with server data
    """
    np.random.seed(random_seed)
    
    # Generate server locations with some clustering (more realistic)
    # Create clusters of servers in different regions
    num_clusters = max(1, num_servers // 3)
    cluster_centers = np.random.uniform(
        low=[x_range[0], y_range[0]], 
        high=[x_range[1], y_range[1]], 
        size=(num_clusters, 2)
    )
    
    server_locations = []
    for _ in range(num_servers):
        # Choose a random cluster center
        center = cluster_centers[np.random.randint(0, num_clusters)]
        # Add some noise around the center
        location = center + np.random.normal(0, 5, 2)
        # Ensure within bounds
        location[0] = np.clip(location[0], x_range[0], x_range[1])
        location[1] = np.clip(location[1], y_range[0], y_range[1])
        server_locations.append(location)
    
    server_locations = np.array(server_locations)
    
    # Generate other server properties
    setup_costs = np.random.uniform(setup_cost_range[0], setup_cost_range[1], num_servers)
    capacities = np.random.randint(capacity_range[0], capacity_range[1], num_servers)
    
    # Create DataFrame
    servers_data = {
        'server_id': [f'S{i+1:03d}' for i in range(num_servers)],
        'location_x': server_locations[:, 0],
        'location_y': server_locations[:, 1],
        'setup_cost': setup_costs,
        'capacity_streams': capacities
    }
    
    return pd.DataFrame(servers_data)


def generate_dmas_data(
    num_dmas: int,
    x_range: Tuple[float, float] = (0, 100),
    y_range: Tuple[float, float] = (0, 100),
    demand_range: Tuple[int, int] = (1, 10),
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate sample DMA data.
    
    Args:
        num_dmas: Number of DMAs to generate
        x_range: Range for X coordinates
        y_range: Range for Y coordinates
        demand_range: Range for DMA demands
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with DMA data
    """
    np.random.seed(random_seed)
    
    # Generate DMA locations with population-based clustering
    # More DMAs in urban areas (higher density regions)
    num_urban_centers = max(1, num_dmas // 10)
    urban_centers = np.random.uniform(
        low=[x_range[0], y_range[0]], 
        high=[x_range[1], y_range[1]], 
        size=(num_urban_centers, 2)
    )
    
    dma_locations = []
    for _ in range(num_dmas):
        if np.random.random() < 0.7:  # 70% in urban areas
            # Choose a random urban center
            center = urban_centers[np.random.randint(0, num_urban_centers)]
            # Add small noise for urban clustering
            location = center + np.random.normal(0, 3, 2)
        else:
            # 30% in rural areas (more spread out)
            location = np.random.uniform(
                low=[x_range[0], y_range[0]], 
                high=[x_range[1], y_range[1]], 
                size=2
            )
        
        # Ensure within bounds
        location[0] = np.clip(location[0], x_range[0], x_range[1])
        location[1] = np.clip(location[1], y_range[0], y_range[1])
        dma_locations.append(location)
    
    dma_locations = np.array(dma_locations)
    
    # Generate demands (higher in urban areas)
    demands = []
    for i, location in enumerate(dma_locations):
        # Find nearest urban center
        distances = np.linalg.norm(urban_centers - location, axis=1)
        min_distance = np.min(distances)
        
        # Higher demand for DMAs closer to urban centers
        if min_distance < 5:  # Urban
            demand = np.random.randint(demand_range[0] + 2, demand_range[1] + 2)
        elif min_distance < 15:  # Suburban
            demand = np.random.randint(demand_range[0] + 1, demand_range[1])
        else:  # Rural
            demand = np.random.randint(demand_range[0], demand_range[1] - 1)
        
        demands.append(max(1, demand))  # Ensure minimum demand of 1
    
    # Create DataFrame
    dmas_data = {
        'dma_id': [f'D{i+1:03d}' for i in range(num_dmas)],
        'location_x': dma_locations[:, 0],
        'location_y': dma_locations[:, 1],
        'demand_streams': demands
    }
    
    return pd.DataFrame(dmas_data)


def validate_data_feasibility(
    servers_df: pd.DataFrame, 
    dmas_df: pd.DataFrame
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that the generated data is feasible for optimization.
    
    Args:
        servers_df: Server data DataFrame
        dmas_df: DMA data DataFrame
        
    Returns:
        Tuple of (is_feasible, validation_info)
    """
    total_capacity = servers_df['capacity_streams'].sum()
    total_demand = dmas_df['demand_streams'].sum()
    
    is_feasible = total_capacity >= total_demand
    
    validation_info = {
        'total_capacity': total_capacity,
        'total_demand': total_demand,
        'capacity_surplus': total_capacity - total_demand,
        'utilization_rate': total_demand / total_capacity if total_capacity > 0 else 0,
        'num_servers': len(servers_df),
        'num_dmas': len(dmas_df)
    }
    
    return is_feasible, validation_info


def generate_sample_data(
    num_servers: int = 5,
    num_dmas: int = 20,
    output_dir: str = "data",
    random_seed: int = 42,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Generate complete sample dataset.
    
    Args:
        num_servers: Number of servers to generate
        num_dmas: Number of DMAs to generate
        output_dir: Directory to save CSV files
        random_seed: Random seed for reproducibility
        validate: Whether to validate data feasibility
        
    Returns:
        Dictionary with generation results and validation info
    """
    logger.info(f"Generating sample data: {num_servers} servers, {num_dmas} DMAs")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    servers_df = generate_servers_data(num_servers, random_seed=random_seed)
    dmas_df = generate_dmas_data(num_dmas, random_seed=random_seed)
    
    # Save to CSV files
    servers_file = output_path / "servers_sample.csv"
    dmas_file = output_path / "dmas_sample.csv"
    
    servers_df.to_csv(servers_file, index=False)
    dmas_df.to_csv(dmas_file, index=False)
    
    logger.info(f"Data saved to {servers_file} and {dmas_file}")
    
    # Validate if requested
    validation_info = {}
    if validate:
        is_feasible, validation_info = validate_data_feasibility(servers_df, dmas_df)
        
        if is_feasible:
            logger.info("✅ Generated data is feasible for optimization")
        else:
            logger.warning("⚠️ Generated data may not be feasible for optimization")
            logger.warning(f"Total demand ({validation_info['total_demand']}) > "
                         f"Total capacity ({validation_info['total_capacity']})")
    
    # Return results
    results = {
        'servers_file': str(servers_file),
        'dmas_file': str(dmas_file),
        'num_servers': num_servers,
        'num_dmas': num_dmas,
        'random_seed': random_seed,
        'validation': validation_info
    }
    
    return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate sample data for Ad-Pod Stitching Server Optimization"
    )
    
    parser.add_argument(
        '--servers', '-s',
        type=int,
        default=5,
        help='Number of servers to generate (default: 5)'
    )
    
    parser.add_argument(
        '--dmas', '-d',
        type=int,
        default=20,
        help='Number of DMAs to generate (default: 20)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='data',
        help='Output directory for CSV files (default: data)'
    )
    
    parser.add_argument(
        '--seed', '-r',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip data feasibility validation'
    )
    
    args = parser.parse_args()
    
    try:
        results = generate_sample_data(
            num_servers=args.servers,
            num_dmas=args.dmas,
            output_dir=args.output_dir,
            random_seed=args.seed,
            validate=not args.no_validate
        )
        
        print(f"\n✅ Sample data generated successfully!")
        print(f"📁 Files saved to: {results['servers_file']} and {results['dmas_file']}")
        print(f"📊 Dataset: {results['num_servers']} servers, {results['num_dmas']} DMAs")
        print(f"🎲 Random seed: {results['random_seed']}")
        
        if results['validation']:
            print(f"\n🔍 Validation Results:")
            print(f"   Total capacity: {results['validation']['total_capacity']}")
            print(f"   Total demand: {results['validation']['total_demand']}")
            print(f"   Capacity surplus: {results['validation']['capacity_surplus']}")
            print(f"   Utilization rate: {results['validation']['utilization_rate']:.2%}")
        
    except Exception as e:
        logger.error(f"Failed to generate sample data: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
